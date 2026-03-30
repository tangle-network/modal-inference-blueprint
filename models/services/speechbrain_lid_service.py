"""
SpeechBrain Language Identification Service - 107-language spoken language recognition.

Capabilities:
- Identify spoken language from audio (107 languages)
- ECAPA-TDNN architecture (VoxLingua107 dataset)
- Language embeddings for downstream tasks
- 6.7% error rate on VoxLingua107 dev set
- Apache 2.0 license

Deploy: modal deploy modal/speechbrain_lid_service.py

Requirements:
- CPU or T4 GPU (lightweight model)
"""

import modal
import os

from base_service import create_modal_app, create_model_volume, download_audio_to_tempfile

# --- Modal resources ---------------------------------------------------------

app = create_modal_app("speechbrain-lid")
model_cache = create_model_volume()

MODEL_CACHE_DIR = "/models"
SB_CACHE = f"{MODEL_CACHE_DIR}/speechbrain"

# --- 107 supported languages -------------------------------------------------

SUPPORTED_LANGUAGES = [
    "ab", "af", "am", "ar", "as", "az", "ba", "be", "bg", "bi",
    "bn", "bo", "br", "bs", "ca", "ceb", "cs", "cy", "da", "de",
    "el", "en", "eo", "es", "et", "eu", "fa", "fi", "fo", "fr",
    "gl", "gn", "gu", "ha", "haw", "he", "hi", "hr", "ht", "hu",
    "hy", "ia", "id", "is", "it", "ja", "jv", "ka", "kk", "km",
    "kn", "ko", "la", "lb", "ln", "lo", "lt", "lv", "mg", "mi",
    "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "nn",
    "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sco",
    "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv",
    "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk",
    "ur", "uz", "vi", "war", "yi", "yo", "zh",
]

# --- Download model at image build time --------------------------------------

def download_speechbrain_model():
    """Download SpeechBrain language ID model during image build."""
    os.makedirs(SB_CACHE, exist_ok=True)

    from speechbrain.inference.classifiers import EncoderClassifier

    print("Downloading speechbrain/lang-id-voxlingua107-ecapa...")
    EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-voxlingua107-ecapa",
        savedir=f"{SB_CACHE}/lang-id-voxlingua107-ecapa",
    )
    print("SpeechBrain LID model download complete.")


# --- Container image ---------------------------------------------------------

image = (
    base_service_layer(modal.Image.debian_slim(python_version="3.10"))
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.3.1",
        "torchaudio==2.3.1",
        "numpy<2",
        "soundfile",
        "fastapi",
        "uvicorn",
        "speechbrain>=1.0.0",
        "transformers>=4.36.0",
        "huggingface_hub",
    )
    .run_function(download_speechbrain_model)
)


# --- Service class -----------------------------------------------------------

@app.cls(
    image=image,
    gpu="T4",
    timeout=300,
    container_idle_timeout=180,
    volumes={MODEL_CACHE_DIR: model_cache},
    allow_concurrent_inputs=10,
)
class SpeechBrainLIDService:
    """Spoken language identification via SpeechBrain ECAPA-TDNN."""

    @modal.enter()
    def load_model(self):
        """Load language identification model on container start."""
        from speechbrain.inference.classifiers import EncoderClassifier

        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa",
            savedir=f"{SB_CACHE}/lang-id-voxlingua107-ecapa",
            run_opts={"device": "cuda"},
        )
        print("SpeechBrain LID model loaded on GPU.")

    @modal.asgi_app()
    def web_app(self):
        """FastAPI app with language identification endpoints."""
        import time
        import traceback
        import base64
        import torch
        import torchaudio
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse

        api = FastAPI()
        svc = self

        @api.get("/health")
        def health():
            return {
                "status": "ok",
                "model": "lang-id-voxlingua107-ecapa",
                "framework": "speechbrain",
                "gpu": "T4",
                "capabilities": [
                    "language_identification",
                    "language_embeddings",
                    "107_languages",
                ],
                "num_languages": len(SUPPORTED_LANGUAGES),
            }

        @api.get("/languages")
        def list_languages():
            """List all 107 supported languages (ISO 639-1 codes)."""
            return {
                "languages": SUPPORTED_LANGUAGES,
                "count": len(SUPPORTED_LANGUAGES),
                "model": "speechbrain/lang-id-voxlingua107-ecapa",
                "dataset": "VoxLingua107",
            }

        @api.post("/identify-language")
        async def identify_language(request: Request):
            """Identify the spoken language in an audio recording.

            Accepts JSON:
              - audio_url (str): URL to audio file
              - audio_file (str, base64): raw audio bytes (base64-encoded)
              - top_k (int, optional): number of top predictions to return (default 5)
              - return_embedding (bool, optional): include language embedding vector

            Returns:
              {
                "language": "en",
                "confidence": 0.97,
                "alternatives": [
                  {"language": "en", "confidence": 0.97},
                  {"language": "de", "confidence": 0.01},
                  ...
                ],
                "duration_seconds": 5.2,
                "processing_time": 0.15
              }
            """
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(
                    status_code=400, content={"error": f"Invalid JSON: {e}"}
                )

            audio_url = body.get("audio_url")
            audio_b64 = body.get("audio_file")
            top_k = body.get("top_k", 5)
            return_embedding = body.get("return_embedding", False)

            if not audio_url and not audio_b64:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Provide audio_url or audio_file (base64)"},
                )

            temp_path = None
            try:
                audio_bytes = None
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)

                temp_path = download_audio_to_tempfile(
                    audio_url=audio_url,
                    audio_file=audio_bytes,
                    target_sr=16000,
                    mono=True,
                )

                # Load and classify
                signal = svc.classifier.load_audio(temp_path)
                prediction = svc.classifier.classify_batch(signal)

                # prediction returns (posterior, score, index, text_label)
                posterior = prediction[0]  # log-softmax probabilities
                probs = torch.exp(posterior).squeeze()

                # Get top-k predictions
                top_values, top_indices = torch.topk(probs, min(top_k, len(probs)))

                # Map indices to language labels
                label_encoder = svc.classifier.hparams.label_encoder
                alternatives = []
                for val, idx in zip(top_values.tolist(), top_indices.tolist()):
                    lang_code = label_encoder.decode_ndim(idx)
                    alternatives.append({
                        "language": lang_code,
                        "confidence": round(val, 4),
                    })

                top_lang = alternatives[0]["language"]
                top_conf = alternatives[0]["confidence"]

                info_obj = torchaudio.info(temp_path)
                duration = info_obj.num_frames / info_obj.sample_rate
                elapsed = time.time() - start_time

                result = {
                    "language": top_lang,
                    "confidence": top_conf,
                    "alternatives": alternatives,
                    "duration_seconds": round(duration, 3),
                    "processing_time": round(elapsed, 3),
                }

                if return_embedding:
                    emb = svc.classifier.encode_batch(signal)
                    result["embedding"] = emb.squeeze().tolist()
                    result["embedding_dimension"] = emb.shape[-1]

                print(
                    f"speechbrain-lid: identified {top_lang} "
                    f"(conf={top_conf:.3f}) from {duration:.1f}s audio "
                    f"in {elapsed:.3f}s"
                )

                return result

            except Exception as e:
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)

        return api


# --- Local entrypoint --------------------------------------------------------

@app.local_entrypoint()
def main():
    print("SpeechBrain Language Identification Service ready")
    print("Deploy with: modal deploy modal/speechbrain_lid_service.py")
