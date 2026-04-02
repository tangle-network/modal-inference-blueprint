use blueprint_std::collections::HashMap;
use blueprint_std::path::PathBuf;
use blueprint_std::sync::Arc;
use blueprint_std::time::Duration;

use alloy::{
    network::EthereumWallet,
    primitives::{keccak256, Address, FixedBytes, B256, U256},
    providers::{Provider, ProviderBuilder},
    signers::local::PrivateKeySigner,
    sol,
    sol_types::SolValue,
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::config::OperatorConfig;

// ---------------------------------------------------------------------------
// ShieldedCredits contract bindings
// ---------------------------------------------------------------------------

sol! {
    #[sol(rpc)]
    interface IShieldedCredits {
        struct SpendAuth {
            bytes32 commitment;
            uint64 serviceId;
            uint8 jobIndex;
            uint256 amount;
            address operator;
            uint256 nonce;
            uint64 expiry;
            bytes signature;
        }

        function authorizeSpend(SpendAuth calldata auth) external returns (bytes32 authHash);
        function claimPayment(bytes32 authHash, address recipient) external;
        function getAccount(bytes32 commitment) external view returns (
            address spendingKey,
            address token,
            uint256 balance,
            uint256 totalFunded,
            uint256 totalSpent,
            uint256 nonce
        );
    }
}

// ---------------------------------------------------------------------------
// SpendAuth payload (decoded from X-Payment-Signature header or request body)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SpendAuthPayload {
    pub commitment: String,
    pub service_id: u64,
    pub job_index: u8,
    pub amount: String,
    pub operator: String,
    pub nonce: u64,
    pub expiry: u64,
    pub signature: String,
}

// ---------------------------------------------------------------------------
// Task-type cost calculation
// ---------------------------------------------------------------------------

/// Usage units consumed by a single request, keyed by task type.
#[derive(Debug, Clone, Default)]
pub struct UsageUnits {
    /// Characters processed (TTS)
    pub characters: u64,
    /// Audio seconds processed * 100 for fixed-point precision (STT/diarize)
    pub audio_centiseconds: u64,
    /// Images generated
    pub images: u64,
    /// Input tokens (text generation)
    pub prompt_tokens: u32,
    /// Output tokens (text generation)
    pub completion_tokens: u32,
}

// ---------------------------------------------------------------------------
// Nonce replay protection
// ---------------------------------------------------------------------------

type NonceKey = (String, u64);

#[derive(Serialize, Deserialize)]
struct NonceRecord {
    commitment: String,
    nonce: u64,
    expiry: u64,
}

/// Replay-protection nonce store with optional file persistence.
///
/// Without persistence (`nonce_store_path` unset), operator restarts clear all
/// nonces, allowing replay of any unexpired SpendAuth signatures.
pub struct NonceStore {
    nonces: RwLock<HashMap<NonceKey, u64>>,
    path: Option<PathBuf>,
}

impl NonceStore {
    /// Create a new nonce store, loading persisted nonces from disk if path is set.
    pub fn load(path: Option<PathBuf>) -> Self {
        let nonces: HashMap<NonceKey, u64> = path
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|data| serde_json::from_str::<Vec<NonceRecord>>(&data).ok())
            .map(|records| {
                records
                    .into_iter()
                    .map(|r| ((r.commitment, r.nonce), r.expiry))
                    .collect()
            })
            .unwrap_or_default();

        if path.is_some() {
            tracing::info!(
                path = ?path,
                count = nonces.len(),
                "NonceStore loaded from disk"
            );
        }

        Self {
            nonces: RwLock::new(nonces),
            path,
        }
    }

    /// Returns true if this (commitment, nonce) pair has already been used.
    pub async fn check_replay(&self, key: &NonceKey, _clock_skew_secs: u64) -> bool {
        let nonces = self.nonces.read().await;
        nonces.contains_key(key)
    }

    /// Mark a nonce as used and persist to disk.
    pub async fn insert(&self, key: NonceKey, expiry: u64, _clock_skew_secs: u64) {
        {
            let mut nonces = self.nonces.write().await;
            nonces.insert(key, expiry);
        }
        self.persist().await;
    }

    async fn persist(&self) {
        let Some(ref path) = self.path else { return };
        let nonces = self.nonces.read().await;
        let records: Vec<NonceRecord> = nonces
            .iter()
            .map(|((commitment, nonce), expiry)| NonceRecord {
                commitment: commitment.clone(),
                nonce: *nonce,
                expiry: *expiry,
            })
            .collect();
        drop(nonces);

        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match serde_json::to_string(&records) {
            Ok(json) => {
                if let Err(e) = std::fs::write(path, json) {
                    tracing::warn!(error = %e, path = ?path, "NonceStore persist failed");
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "NonceStore serialize failed");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// On-chain account info
// ---------------------------------------------------------------------------

pub struct AccountInfo {
    pub spending_key: Address,
    pub balance: U256,
}

// ---------------------------------------------------------------------------
// BillingClient
// ---------------------------------------------------------------------------

/// Handles ShieldedCredits billing operations.
pub struct BillingClient {
    config: Arc<OperatorConfig>,
    wallet: EthereumWallet,
    shielded_credits: Address,
    rpc_url: reqwest::Url,
    /// This operator's Ethereum address, derived from operator_key.
    operator_address: Address,
}

impl BillingClient {
    pub async fn new(config: Arc<OperatorConfig>) -> anyhow::Result<Self> {
        let key = &config.billing.tangle.operator_key;
        let signer: PrivateKeySigner = key.parse()?;
        let operator_address = signer.address();

        // Block plaintext keys in production mode.
        if std::env::var("PRODUCTION").unwrap_or_default() == "1" {
            anyhow::bail!(
                "PRODUCTION=1 but operator_key is loaded from plaintext. \
                 Use a KMS (AWS KMS, HashiCorp Vault) or encrypted keystore. \
                 Unset PRODUCTION to run in dev mode."
            );
        }
        tracing::warn!(
            "operator_key loaded from plaintext — \
             use a KMS or encrypted keystore in production (set PRODUCTION=1 to enforce)"
        );

        let wallet = EthereumWallet::from(signer);
        let shielded_credits: Address = config.billing.tangle.shielded_credits.parse()?;
        let rpc_url: reqwest::Url = config.billing.tangle.rpc_url.parse()?;

        Ok(Self {
            config,
            wallet,
            shielded_credits,
            rpc_url,
            operator_address,
        })
    }

    /// Returns the operator's Ethereum address.
    pub fn operator_address(&self) -> Address {
        self.operator_address
    }

    /// Calculate cost in base token units for the given usage.
    ///
    /// Maps task types to the appropriate pricing dimension:
    /// - TTS/clone/voice-design: per 1K characters
    /// - STT/diarize/s2s/vad: per second of audio
    /// - Video (gen/avatar/lipsync/understanding): per second of video
    /// - Image generation: per image
    /// - Text generation: per million tokens (input + output)
    /// - Music: per second of music
    /// - Fixed jobs (stitch/enhance/convert): per job
    pub fn calculate_cost(&self, task_type: &str, usage: &UsageUnits) -> u64 {
        let p = &self.config.billing.pricing;
        match task_type {
            // Per 1K characters
            "tts" | "clone" | "voice-design" => {
                usage.characters.saturating_mul(p.price_per_1k_chars) / 1000
            }
            // Per second audio
            "stt" | "diarize" | "translate" | "langid" | "vad" | "s2s" | "speakerid" => {
                let seconds = usage.audio_centiseconds / 100;
                seconds.saturating_mul(p.price_per_second_audio)
            }
            // Per second video
            "video-generation" | "video-avatar" | "video-lipsync" | "video-understanding" => {
                let seconds = usage.audio_centiseconds / 100; // reuse field for video duration
                seconds.saturating_mul(p.price_per_second_video)
            }
            // Per image
            "image-generation" | "image" => {
                usage.images.saturating_mul(p.price_per_image)
            }
            // Per second music
            "music-generation" | "music" => {
                let seconds = usage.audio_centiseconds / 100;
                seconds.saturating_mul(p.price_per_second_music)
            }
            // Per job (fixed cost)
            "video-stitch" | "audio-processing" | "voice-conversion" | "enhance" => {
                p.price_per_job
            }
            // Default: per million tokens (text generation, embedding, rerank)
            _ => {
                let input_cost = (usage.prompt_tokens as u64)
                    .saturating_mul(p.price_per_million_input_tokens) / 1_000_000;
                let output_cost = (usage.completion_tokens as u64)
                    .saturating_mul(p.price_per_million_output_tokens) / 1_000_000;
                input_cost.saturating_add(output_cost)
            }
        }
    }

    fn build_auth(
        &self,
        spend_auth: &SpendAuthPayload,
    ) -> anyhow::Result<IShieldedCredits::SpendAuth> {
        let commitment: B256 = spend_auth.commitment.parse()?;
        let amount: U256 = spend_auth.amount.parse()?;
        let operator: Address = spend_auth.operator.parse()?;
        let sig_bytes = hex::decode(
            spend_auth
                .signature
                .strip_prefix("0x")
                .unwrap_or(&spend_auth.signature),
        )?;

        Ok(IShieldedCredits::SpendAuth {
            commitment: FixedBytes(commitment.0),
            serviceId: spend_auth.service_id,
            jobIndex: spend_auth.job_index,
            amount,
            operator,
            nonce: U256::from(spend_auth.nonce),
            expiry: spend_auth.expiry,
            signature: sig_bytes.into(),
        })
    }

    fn auth_hash(spend_auth: &SpendAuthPayload) -> anyhow::Result<FixedBytes<32>> {
        let commitment: B256 = spend_auth.commitment.parse()?;
        let hash = keccak256(
            (
                FixedBytes::<32>(commitment.0),
                U256::from(spend_auth.service_id),
                U256::from(spend_auth.job_index),
                U256::from(spend_auth.nonce),
            )
                .abi_encode(),
        );
        Ok(FixedBytes(hash.0))
    }

    async fn check_gas_price(&self) -> anyhow::Result<()> {
        let max_gwei = self.config.billing.max_gas_price_gwei;
        if max_gwei == 0 {
            return Ok(());
        }

        let provider = ProviderBuilder::new().connect_http(self.rpc_url.clone());
        let gas_price = provider.get_gas_price().await?;
        let gas_price_gwei = gas_price / 1_000_000_000;

        if gas_price_gwei > max_gwei as u128 {
            anyhow::bail!(
                "gas price {gas_price_gwei} gwei exceeds cap {max_gwei} gwei — deferring tx"
            );
        }

        Ok(())
    }

    /// Pre-authorize spending on-chain. Must be called before proxying to Modal.
    pub async fn authorize_spend(&self, spend_auth: &SpendAuthPayload) -> anyhow::Result<()> {
        self.check_gas_price().await?;

        let auth = self.build_auth(spend_auth)?;

        let provider = ProviderBuilder::new()
            .wallet(self.wallet.clone())
            .connect_http(self.rpc_url.clone());

        let contract = IShieldedCredits::new(self.shielded_credits, &provider);

        let pending = contract.authorizeSpend(auth).send().await?;
        let receipt = pending.get_receipt().await?;
        tracing::info!(
            tx_hash = %receipt.transaction_hash,
            "authorizeSpend confirmed"
        );

        Ok(())
    }

    /// Claim payment on-chain after inference is served.
    ///
    /// IMPORTANT: The ShieldedCredits contract `claimPayment(bytes32, address)`
    /// always settles the full pre-authorized amount. There is no partial
    /// settlement. The `actual_amount` parameter is logged for auditing only.
    ///
    /// Retries up to `claim_max_retries` times on failure with exponential backoff.
    pub async fn claim_payment(
        &self,
        spend_auth: &SpendAuthPayload,
        actual_amount: u64,
    ) -> anyhow::Result<()> {
        let auth_hash = Self::auth_hash(spend_auth)?;
        let operator: Address = spend_auth.operator.parse()?;
        let max_retries = self.config.billing.claim_max_retries;

        tracing::info!(
            actual_amount = actual_amount,
            preauth_amount = %spend_auth.amount,
            "claiming payment (actual metered cost)"
        );

        let mut last_err = None;
        for attempt in 0..=max_retries {
            if let Err(e) = self.check_gas_price().await {
                tracing::warn!(error = %e, attempt, "gas price check failed for claimPayment");
                last_err = Some(e);
                let delay = Duration::from_millis(500 * 2u64.pow(attempt));
                tokio::time::sleep(delay).await;
                continue;
            }
            if attempt > 0 {
                let delay = Duration::from_millis(500 * 2u64.pow(attempt - 1));
                tracing::warn!(
                    attempt,
                    delay_ms = delay.as_millis() as u64,
                    "retrying claimPayment"
                );
                tokio::time::sleep(delay).await;
            }

            let provider = ProviderBuilder::new()
                .wallet(self.wallet.clone())
                .connect_http(self.rpc_url.clone());

            let contract = IShieldedCredits::new(self.shielded_credits, &provider);

            match contract.claimPayment(auth_hash, operator).send().await {
                Ok(pending) => match pending.get_receipt().await {
                    Ok(receipt) => {
                        tracing::info!(
                            tx_hash = %receipt.transaction_hash,
                            actual_amount = actual_amount,
                            attempt,
                            "claimPayment confirmed"
                        );
                        return Ok(());
                    }
                    Err(e) => {
                        last_err = Some(e.into());
                    }
                },
                Err(e) => {
                    last_err = Some(e.into());
                }
            }
        }

        let err = last_err.unwrap_or_else(|| anyhow::anyhow!("claimPayment failed"));
        tracing::error!(
            error = %err,
            auth_hash = %auth_hash,
            actual_amount,
            commitment = %spend_auth.commitment,
            "claimPayment FAILED after {} retries — operator served inference for free. Manual recovery required.",
            max_retries
        );
        Err(err)
    }

    /// Query on-chain account info (spending key + balance) for a ShieldedCredits account.
    pub async fn get_account_info(&self, commitment: &str) -> anyhow::Result<AccountInfo> {
        let commitment: B256 = commitment.parse()?;

        let provider = ProviderBuilder::new().connect_http(self.rpc_url.clone());

        let contract = IShieldedCredits::new(self.shielded_credits, &provider);

        let result = contract.getAccount(FixedBytes(commitment.0)).call().await?;
        Ok(AccountInfo {
            spending_key: result.spendingKey,
            balance: result.balance,
        })
    }
}

// ---------------------------------------------------------------------------
// EIP-712 signature verification
// ---------------------------------------------------------------------------

/// Recover the signer address from a SpendAuth EIP-712 signature.
///
/// Returns the recovered Ethereum address on success. The caller MUST compare
/// this against the account's on-chain spending key to authenticate the request.
///
/// Also checks expiry with the given clock skew tolerance.
pub fn recover_spend_auth_signer(
    auth: &SpendAuthPayload,
    shielded_credits_addr: &str,
    chain_id: u64,
    clock_skew_tolerance_secs: u64,
) -> Result<Address, String> {
    use k256::ecdsa::{RecoveryId, Signature, VerifyingKey};

    let shielded_addr: Address = shielded_credits_addr
        .parse()
        .map_err(|e| format!("invalid shielded_credits address: {e}"))?;

    let domain_separator = keccak256(
        (
            keccak256(
                b"EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)",
            ),
            keccak256(b"ShieldedCredits"),
            keccak256(b"1"),
            U256::from(chain_id),
            shielded_addr,
        )
            .abi_encode(),
    );

    let spend_typehash = keccak256(
        b"SpendAuthorization(bytes32 commitment,uint64 serviceId,uint8 jobIndex,uint256 amount,address operator,uint256 nonce,uint64 expiry)",
    );

    let commitment: B256 = auth
        .commitment
        .parse()
        .map_err(|e| format!("invalid commitment: {e}"))?;
    let amount: U256 = auth
        .amount
        .parse()
        .map_err(|e| format!("invalid amount: {e}"))?;
    let operator: Address = auth
        .operator
        .parse()
        .map_err(|e| format!("invalid operator address: {e}"))?;

    let struct_hash = keccak256(
        (
            spend_typehash,
            commitment,
            U256::from(auth.service_id),
            U256::from(auth.job_index),
            amount,
            operator,
            U256::from(auth.nonce),
            U256::from(auth.expiry),
        )
            .abi_encode(),
    );

    let digest = keccak256(
        [
            &[0x19, 0x01],
            domain_separator.as_slice(),
            struct_hash.as_slice(),
        ]
        .concat(),
    );

    let sig_hex = auth.signature.strip_prefix("0x").unwrap_or(&auth.signature);
    let sig_bytes = hex::decode(sig_hex).map_err(|e| format!("invalid signature hex: {e}"))?;
    if sig_bytes.len() != 65 {
        return Err(format!(
            "invalid signature length: expected 65, got {}",
            sig_bytes.len()
        ));
    }

    let v = sig_bytes[64];
    let recovery_id = match v {
        27 => 0u8,
        28 => 1u8,
        0 | 1 => v,
        _ => return Err(format!("invalid signature recovery byte: {v}")),
    };

    let signature =
        Signature::from_slice(&sig_bytes[..64]).map_err(|e| format!("invalid signature: {e}"))?;
    let rid = RecoveryId::try_from(recovery_id).map_err(|e| format!("invalid recovery id: {e}"))?;
    let recovered = VerifyingKey::recover_from_prehash(digest.as_slice(), &signature, rid)
        .map_err(|e| format!("ecrecover failed: {e}"))?;

    // Convert recovered public key to Ethereum address
    let pubkey_bytes = recovered.to_encoded_point(false);
    let pubkey_hash = keccak256(&pubkey_bytes.as_bytes()[1..]);
    let recovered_address = Address::from_slice(&pubkey_hash[12..]);

    // Check expiry with clock skew tolerance
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|_| "system clock is before UNIX epoch".to_string())?
        .as_secs();
    if now > auth.expiry.saturating_add(clock_skew_tolerance_secs) {
        return Err(format!(
            "SpendAuth expired: now={now}, expiry={}, tolerance={clock_skew_tolerance_secs}s",
            auth.expiry
        ));
    }

    Ok(recovered_address)
}
