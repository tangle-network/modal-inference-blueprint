"""
PersonaPlex Voice Encoder

Creates custom voice embeddings from audio samples for PersonaPlex voice cloning.
Uses the Kyutai Mimi codec to encode audio into tokens that capture speaker identity.

The Mimi codec encodes audio into discrete tokens across 8 codebooks, which PersonaPlex
uses to capture speaker characteristics for voice cloning.

Usage:
  from personaplex_voice_encoder import VoiceEncoder

  encoder = VoiceEncoder()
  embedding = encoder.encode_voice("sample.wav")
  encoder.save_embedding(embedding, "custom_voice.pt")
"""

import io
import torch
import torchaudio
from pathlib import Path
from typing import Union, Optional, Tuple
from dataclasses import dataclass


# Constants
TARGET_SAMPLE_RATE = 24000
NUM_CODEBOOKS = 8


@dataclass
class VoiceEmbedding:
    """Voice embedding for PersonaPlex.

    Attributes:
        codes: Mimi-encoded audio tokens with shape [1, K=8, T] where K is the
               number of codebooks and T is the number of time steps.
        duration_seconds: Duration of the original audio in seconds.
        sample_rate: Sample rate of the audio (always 24000 for Mimi).
    """
    codes: torch.Tensor  # [1, K=8, T] audio tokens
    duration_seconds: float
    sample_rate: int = TARGET_SAMPLE_RATE

    def __post_init__(self):
        """Validate embedding shape."""
        if self.codes.dim() != 3:
            raise ValueError(f"Expected 3D tensor [B, K, T], got {self.codes.dim()}D")
        if self.codes.shape[1] != NUM_CODEBOOKS:
            raise ValueError(f"Expected {NUM_CODEBOOKS} codebooks, got {self.codes.shape[1]}")


def resample_audio(
    waveform: torch.Tensor,
    orig_sr: int,
    target_sr: int = TARGET_SAMPLE_RATE,
) -> torch.Tensor:
    """Resample audio to target sample rate.

    Args:
        waveform: Audio tensor of shape [channels, samples] or [samples].
        orig_sr: Original sample rate.
        target_sr: Target sample rate (default 24000 for Mimi).

    Returns:
        Resampled audio tensor.
    """
    if orig_sr == target_sr:
        return waveform

    resampler = torchaudio.transforms.Resample(
        orig_freq=orig_sr,
        new_freq=target_sr,
    )
    return resampler(waveform)


def normalize_audio(
    waveform: torch.Tensor,
    target_db: float = -3.0,
) -> torch.Tensor:
    """Normalize audio volume to target dB level.

    Uses peak normalization to scale audio to the target dB level below 0 dB.

    Args:
        waveform: Audio tensor of shape [channels, samples] or [samples].
        target_db: Target peak level in dB (default -3.0 dB).

    Returns:
        Normalized audio tensor.
    """
    # Find peak amplitude
    peak = waveform.abs().max()

    if peak == 0:
        return waveform

    # Calculate target amplitude from dB
    target_amplitude = 10 ** (target_db / 20)

    # Scale to target
    return waveform * (target_amplitude / peak)


def trim_silence(
    waveform: torch.Tensor,
    threshold: float = 0.01,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    """Trim silence from start and end of audio.

    Uses energy-based voice activity detection to find the first and last
    frames with energy above the threshold.

    Args:
        waveform: Audio tensor of shape [channels, samples] or [samples].
        threshold: Energy threshold for silence detection (0.0 to 1.0).
        frame_length: Length of each analysis frame.
        hop_length: Number of samples between frames.

    Returns:
        Trimmed audio tensor.
    """
    # Ensure 2D tensor [channels, samples]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Convert to mono for analysis
    mono = waveform.mean(dim=0)

    # Calculate frame energies
    num_samples = mono.shape[0]
    num_frames = (num_samples - frame_length) // hop_length + 1

    if num_frames <= 0:
        return waveform

    # Calculate RMS energy for each frame
    energies = []
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = mono[start:end]
        rms = torch.sqrt(torch.mean(frame ** 2))
        energies.append(rms.item())

    # Normalize energies
    max_energy = max(energies) if energies else 1.0
    if max_energy > 0:
        energies = [e / max_energy for e in energies]

    # Find first and last frames above threshold
    start_frame = 0
    end_frame = len(energies) - 1

    for i, energy in enumerate(energies):
        if energy > threshold:
            start_frame = i
            break

    for i in range(len(energies) - 1, -1, -1):
        if energies[i] > threshold:
            end_frame = i
            break

    # Convert frames back to samples
    start_sample = start_frame * hop_length
    end_sample = min((end_frame + 1) * hop_length + frame_length, num_samples)

    return waveform[:, start_sample:end_sample]


def convert_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Convert audio to mono by averaging channels.

    Args:
        waveform: Audio tensor of shape [channels, samples] or [samples].

    Returns:
        Mono audio tensor of shape [1, samples].
    """
    if waveform.dim() == 1:
        return waveform.unsqueeze(0)

    if waveform.shape[0] == 1:
        return waveform

    # Average channels to mono
    return waveform.mean(dim=0, keepdim=True)


class VoiceEncoder:
    """Encodes audio into PersonaPlex-compatible voice embeddings using Mimi codec.

    The encoder uses Kyutai's Mimi codec to convert audio into discrete tokens
    across 8 codebooks. These tokens capture the speaker's voice characteristics
    and can be used by PersonaPlex for voice cloning.

    Attributes:
        device: The device to run inference on (cuda, cpu, or mps).
        mimi: The loaded Mimi encoder model.
    """

    def __init__(self, device: Optional[str] = None):
        """Initialize the VoiceEncoder with Mimi codec.

        Args:
            device: Device to run on. If None, automatically selects cuda if
                    available, then mps, then cpu.

        Raises:
            RuntimeError: If Mimi model fails to load.
        """
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self._mimi = None

    @property
    def mimi(self):
        """Lazily load Mimi encoder on first access."""
        if self._mimi is None:
            self._mimi = self._load_mimi()
        return self._mimi

    def _load_mimi(self):
        """Load the Mimi encoder model.

        Returns:
            Loaded Mimi model configured for 8 codebooks.

        Raises:
            RuntimeError: If model fails to load.
        """
        try:
            from huggingface_hub import hf_hub_download
            from moshi.models import loaders

            # Download and load Mimi weights
            mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
            mimi = loaders.get_mimi(mimi_weight, device=self.device)
            mimi.set_num_codebooks(NUM_CODEBOOKS)
            mimi.eval()

            return mimi

        except ImportError as e:
            raise RuntimeError(
                "Failed to import moshi. Install with: pip install moshi"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load Mimi encoder: {e}") from e

    def _load_audio(self, audio_path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
        """Load audio file.

        Args:
            audio_path: Path to audio file.

        Returns:
            Tuple of (waveform, sample_rate).

        Raises:
            FileNotFoundError: If audio file doesn't exist.
            RuntimeError: If audio fails to load.
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            waveform, sample_rate = torchaudio.load(str(audio_path))
            return waveform, sample_rate
        except Exception as e:
            raise RuntimeError(f"Failed to load audio from {audio_path}: {e}") from e

    def _preprocess_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        max_duration_seconds: float,
        trim_silence_enabled: bool = True,
        normalize_enabled: bool = True,
    ) -> torch.Tensor:
        """Preprocess audio for Mimi encoding.

        Applies the following processing steps:
        1. Convert to mono
        2. Resample to 24kHz
        3. Trim silence (optional)
        4. Normalize volume (optional)
        5. Truncate to max duration
        6. Add batch dimension

        Args:
            waveform: Raw audio tensor.
            sample_rate: Original sample rate.
            max_duration_seconds: Maximum duration to keep.
            trim_silence_enabled: Whether to trim silence.
            normalize_enabled: Whether to normalize volume.

        Returns:
            Preprocessed audio tensor of shape [1, 1, samples].
        """
        # Convert to mono
        waveform = convert_to_mono(waveform)

        # Resample to 24kHz
        waveform = resample_audio(waveform, sample_rate, TARGET_SAMPLE_RATE)

        # Trim silence
        if trim_silence_enabled:
            waveform = trim_silence(waveform)

        # Normalize volume
        if normalize_enabled:
            waveform = normalize_audio(waveform)

        # Truncate to max duration
        max_samples = int(max_duration_seconds * TARGET_SAMPLE_RATE)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]

        # Add batch dimension: [1, samples] -> [1, 1, samples]
        waveform = waveform.unsqueeze(0)

        return waveform

    def encode_voice(
        self,
        audio_path: Union[str, Path],
        max_duration_seconds: float = 10.0,
        trim_silence_enabled: bool = True,
        normalize_enabled: bool = True,
    ) -> VoiceEmbedding:
        """Encode audio file into voice embedding.

        Loads an audio file, preprocesses it (resampling, normalization, etc.),
        and encodes it using the Mimi codec into discrete tokens.

        Args:
            audio_path: Path to audio file. Supports any format that torchaudio
                        can load (wav, mp3, flac, ogg, etc.).
            max_duration_seconds: Maximum audio duration to use (default 10s).
                                 Longer audio will be truncated.
            trim_silence_enabled: Whether to trim silence from start/end.
            normalize_enabled: Whether to normalize audio volume.

        Returns:
            VoiceEmbedding containing Mimi-encoded tokens.

        Raises:
            FileNotFoundError: If audio file doesn't exist.
            RuntimeError: If encoding fails.
        """
        # Load audio
        waveform, sample_rate = self._load_audio(audio_path)

        # Preprocess
        waveform = self._preprocess_audio(
            waveform,
            sample_rate,
            max_duration_seconds,
            trim_silence_enabled,
            normalize_enabled,
        )

        # Calculate duration
        duration_seconds = waveform.shape[-1] / TARGET_SAMPLE_RATE

        # Move to device and encode
        waveform = waveform.to(self.device)

        with torch.no_grad():
            codes = self.mimi.encode(waveform)

        # Move codes to CPU for storage
        codes = codes.cpu()

        return VoiceEmbedding(
            codes=codes,
            duration_seconds=duration_seconds,
            sample_rate=TARGET_SAMPLE_RATE,
        )

    def encode_voice_from_bytes(
        self,
        audio_bytes: bytes,
        format: str = "wav",
        max_duration_seconds: float = 10.0,
        trim_silence_enabled: bool = True,
        normalize_enabled: bool = True,
    ) -> VoiceEmbedding:
        """Encode audio from bytes into voice embedding.

        Useful for processing audio received over network or from memory
        without writing to disk.

        Args:
            audio_bytes: Raw audio bytes.
            format: Audio format hint (wav, mp3, flac, etc.).
            max_duration_seconds: Maximum audio duration to use.
            trim_silence_enabled: Whether to trim silence from start/end.
            normalize_enabled: Whether to normalize audio volume.

        Returns:
            VoiceEmbedding containing Mimi-encoded tokens.

        Raises:
            RuntimeError: If encoding fails.
        """
        # Load from bytes
        try:
            buffer = io.BytesIO(audio_bytes)
            waveform, sample_rate = torchaudio.load(buffer, format=format)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio from bytes: {e}") from e

        # Preprocess
        waveform = self._preprocess_audio(
            waveform,
            sample_rate,
            max_duration_seconds,
            trim_silence_enabled,
            normalize_enabled,
        )

        # Calculate duration
        duration_seconds = waveform.shape[-1] / TARGET_SAMPLE_RATE

        # Move to device and encode
        waveform = waveform.to(self.device)

        with torch.no_grad():
            codes = self.mimi.encode(waveform)

        # Move codes to CPU for storage
        codes = codes.cpu()

        return VoiceEmbedding(
            codes=codes,
            duration_seconds=duration_seconds,
            sample_rate=TARGET_SAMPLE_RATE,
        )

    def encode_voice_from_tensor(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        max_duration_seconds: float = 10.0,
        trim_silence_enabled: bool = True,
        normalize_enabled: bool = True,
    ) -> VoiceEmbedding:
        """Encode audio tensor into voice embedding.

        Useful for processing audio that's already loaded as a tensor.

        Args:
            waveform: Audio tensor of shape [channels, samples] or [samples].
            sample_rate: Sample rate of the audio.
            max_duration_seconds: Maximum audio duration to use.
            trim_silence_enabled: Whether to trim silence from start/end.
            normalize_enabled: Whether to normalize audio volume.

        Returns:
            VoiceEmbedding containing Mimi-encoded tokens.

        Raises:
            RuntimeError: If encoding fails.
        """
        # Preprocess
        waveform = self._preprocess_audio(
            waveform,
            sample_rate,
            max_duration_seconds,
            trim_silence_enabled,
            normalize_enabled,
        )

        # Calculate duration
        duration_seconds = waveform.shape[-1] / TARGET_SAMPLE_RATE

        # Move to device and encode
        waveform = waveform.to(self.device)

        with torch.no_grad():
            codes = self.mimi.encode(waveform)

        # Move codes to CPU for storage
        codes = codes.cpu()

        return VoiceEmbedding(
            codes=codes,
            duration_seconds=duration_seconds,
            sample_rate=TARGET_SAMPLE_RATE,
        )

    @staticmethod
    def save_embedding(
        embedding: VoiceEmbedding,
        path: Union[str, Path],
    ) -> None:
        """Save embedding as .pt file compatible with PersonaPlex.

        Saves the embedding in a format that PersonaPlex can load directly.
        The file contains:
        - codes: The Mimi-encoded audio tokens
        - duration_seconds: Original audio duration
        - sample_rate: Sample rate (24000)
        - num_codebooks: Number of codebooks (8)

        Args:
            embedding: VoiceEmbedding to save.
            path: Output file path (should end in .pt).

        Raises:
            IOError: If file cannot be written.
        """
        path = Path(path)

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save as dictionary for compatibility
        data = {
            "codes": embedding.codes,
            "duration_seconds": embedding.duration_seconds,
            "sample_rate": embedding.sample_rate,
            "num_codebooks": NUM_CODEBOOKS,
        }

        try:
            torch.save(data, path)
        except Exception as e:
            raise IOError(f"Failed to save embedding to {path}: {e}") from e

    @staticmethod
    def load_embedding(path: Union[str, Path]) -> VoiceEmbedding:
        """Load embedding from .pt file.

        Args:
            path: Path to .pt file.

        Returns:
            VoiceEmbedding loaded from file.

        Raises:
            FileNotFoundError: If file doesn't exist.
            RuntimeError: If file format is invalid.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Embedding file not found: {path}")

        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding from {path}: {e}") from e

        # Handle both old format (just tensor) and new format (dict)
        if isinstance(data, torch.Tensor):
            codes = data
            duration_seconds = codes.shape[-1] / 12.5  # Mimi frame rate
        elif isinstance(data, dict):
            codes = data["codes"]
            duration_seconds = data.get("duration_seconds", codes.shape[-1] / 12.5)
        else:
            raise RuntimeError(f"Invalid embedding format in {path}")

        return VoiceEmbedding(
            codes=codes,
            duration_seconds=duration_seconds,
            sample_rate=TARGET_SAMPLE_RATE,
        )


def create_voice_prompt(
    audio_paths: list[Union[str, Path]],
    output_path: Union[str, Path],
    max_total_duration: float = 30.0,
    device: Optional[str] = None,
) -> VoiceEmbedding:
    """Create a voice prompt from multiple audio samples.

    Combines multiple audio samples into a single voice embedding, which
    can improve voice cloning quality by capturing more speaker variation.

    Args:
        audio_paths: List of paths to audio files.
        output_path: Path to save the combined embedding.
        max_total_duration: Maximum total duration across all samples.
        device: Device to run on (auto-detected if None).

    Returns:
        Combined VoiceEmbedding.

    Raises:
        ValueError: If no audio paths provided.
        RuntimeError: If encoding fails.
    """
    if not audio_paths:
        raise ValueError("At least one audio path must be provided")

    encoder = VoiceEncoder(device=device)

    # Calculate per-sample duration
    per_sample_duration = max_total_duration / len(audio_paths)

    # Encode each sample
    all_codes = []
    total_duration = 0.0

    for audio_path in audio_paths:
        embedding = encoder.encode_voice(
            audio_path,
            max_duration_seconds=per_sample_duration,
        )
        all_codes.append(embedding.codes)
        total_duration += embedding.duration_seconds

    # Concatenate codes along time dimension
    combined_codes = torch.cat(all_codes, dim=-1)

    combined_embedding = VoiceEmbedding(
        codes=combined_codes,
        duration_seconds=total_duration,
        sample_rate=TARGET_SAMPLE_RATE,
    )

    # Save
    VoiceEncoder.save_embedding(combined_embedding, output_path)

    return combined_embedding


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Encode audio into PersonaPlex voice embedding"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input audio file path",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output .pt file path",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=10.0,
        help="Maximum audio duration in seconds (default: 10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cpu, mps). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Disable silence trimming",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable volume normalization",
    )

    args = parser.parse_args()

    print(f"Loading encoder on device: {args.device or 'auto'}")
    encoder = VoiceEncoder(device=args.device)

    print(f"Encoding audio: {args.input}")
    embedding = encoder.encode_voice(
        args.input,
        max_duration_seconds=args.max_duration,
        trim_silence_enabled=not args.no_trim,
        normalize_enabled=not args.no_normalize,
    )

    print(f"Encoded {embedding.duration_seconds:.2f}s of audio")
    print(f"Codes shape: {list(embedding.codes.shape)}")

    print(f"Saving to: {args.output}")
    encoder.save_embedding(embedding, args.output)

    print("Done!")
