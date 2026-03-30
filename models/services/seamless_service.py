"""
SeamlessM4T v2 Translation Service - 100+ language S2S/S2T/T2S/T2T translation.

Capabilities:
- Speech-to-speech translation (S2ST)
- Speech-to-text translation (S2TT)
- Text-to-speech translation (T2ST)
- Text-to-text translation (T2TT)
- Automatic speech recognition (ASR)
- 100+ source languages, 35+ target speech languages
- Meta, CC-BY-NC-4.0 license

Deploy: modal deploy modal/seamless_service.py

Requirements:
- transformers >= 4.39
- A10G GPU (model is ~4GB; A100 for batch workloads)
"""

import modal
import os

from base_service import create_modal_app, create_model_volume, download_audio_to_tempfile

# --- Modal resources ---------------------------------------------------------

app = create_modal_app("seamless-m4t")
model_cache = create_model_volume()

MODEL_CACHE_DIR = "/models"
HF_CACHE = f"{MODEL_CACHE_DIR}/huggingface"

# --- Download model at image build time --------------------------------------

def download_seamless():
    """Download SeamlessM4T v2 during image build."""
    os.makedirs(HF_CACHE, exist_ok=True)
    os.environ["HF_HOME"] = HF_CACHE

    from transformers import AutoProcessor, SeamlessM4Tv2Model

    print("Downloading facebook/seamless-m4t-v2-large...")
    AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
    print("SeamlessM4T v2 download complete.")


# --- Container image ---------------------------------------------------------

image = (
    base_service_layer(modal.Image.debian_slim(python_version="3.11"))
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.3.1",
        "torchaudio==2.3.1",
        "numpy<2",
        "soundfile",
        "fastapi",
        "uvicorn",
        "transformers>=4.40",
        "sentencepiece",
        "accelerate",
    )
    .run_function(download_seamless)
)


# --- Service class -----------------------------------------------------------

SAMPLE_RATE = 16000  # SeamlessM4T v2 output sample rate

@app.cls(
    image=image,
    gpu="A10G",
    timeout=600,
    container_idle_timeout=300,
    volumes={MODEL_CACHE_DIR: model_cache},
    allow_concurrent_inputs=5,
)
class SeamlessService:
    """Multi-modal translation via SeamlessM4T v2."""

    @modal.enter()
    def load_model(self):
        """Load SeamlessM4T v2 on container start."""
        import torch
        os.environ["HF_HOME"] = HF_CACHE

        from transformers import AutoProcessor, SeamlessM4Tv2Model

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
        self.model = SeamlessM4Tv2Model.from_pretrained(
            "facebook/seamless-m4t-v2-large",
        ).to(self.device)
        self.model.eval()
        print("SeamlessM4T v2 loaded on GPU.")

    @modal.asgi_app()
    def web_app(self):
        import time
        import traceback
        import base64
        import io
        import soundfile as sf
        from fastapi import FastAPI, Request
        from fastapi.responses import Response, JSONResponse

        api = FastAPI()
        svc = self

        @api.get("/health")
        def health():
            return {
                "status": "ok",
                "model": "seamless-m4t-v2-large",
                "gpu": "A10G",
                "capabilities": ["s2st", "s2tt", "t2st", "t2tt", "asr"],
                "source_languages": "100+",
                "target_speech_languages": "35+",
                "sample_rate": SAMPLE_RATE,
            }

        @api.post("/translate")
        async def translate(request: Request):
            """Translate audio to target language (speech + text).

            Accepts JSON:
              - audio_url (str): URL to audio file
              - audio_file (str, base64): raw audio bytes (base64-encoded)
              - target_language (str): target language code (e.g. "fra", "spa", "deu", "rus", "cmn")
              - source_language (str, optional): source language hint
              - speaker_id (int, optional): speaker voice ID (default 0)
              - generate_speech (bool, optional): also generate translated audio (default true)

            Returns:
              If generate_speech=true: WAV audio with headers:
                X-Translated-Text, X-Target-Language, X-Processing-Time
              If generate_speech=false: JSON {text, target_language, processing_time}
            """
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            audio_url = body.get("audio_url")
            audio_b64 = body.get("audio_file")
            target_language = body.get("target_language")
            speaker_id = body.get("speaker_id", 0)
            generate_speech = body.get("generate_speech", True)

            if not audio_url and not audio_b64:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Provide audio_url or audio_file (base64)"},
                )
            if not target_language:
                return JSONResponse(
                    status_code=400,
                    content={"error": "target_language is required (e.g. 'fra', 'spa', 'deu')"},
                )

            temp_path = None
            try:
                import torch

                audio_bytes = None
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)

                temp_path = download_audio_to_tempfile(
                    audio_url=audio_url,
                    audio_file=audio_bytes,
                    target_sr=SAMPLE_RATE,
                    mono=True,
                )

                audio_array, sr = sf.read(temp_path, dtype="float32")
                duration = len(audio_array) / sr

                audio_inputs = svc.processor(
                    audios=audio_array,
                    sampling_rate=sr,
                    return_tensors="pt",
                ).to(svc.device)

                with torch.no_grad():
                    if generate_speech:
                        output = svc.model.generate(
                            **audio_inputs,
                            tgt_lang=target_language,
                            speaker_id=speaker_id,
                            generate_speech=True,
                            return_intermediate_token_ids=True,
                        )

                        # Extract audio waveform
                        waveform = output[0].cpu().numpy().squeeze()

                        # Extract translated text
                        translated_text = ""
                        if hasattr(output, "sequences") and output.sequences is not None:
                            translated_text = svc.processor.decode(
                                output.sequences[0].tolist(), skip_special_tokens=True,
                            )

                        buf = io.BytesIO()
                        sf.write(buf, waveform, SAMPLE_RATE, format="WAV")
                        buf.seek(0)
                        wav_data = buf.read()

                        elapsed = time.time() - start_time

                        print(
                            f"seamless: translated {duration:.1f}s audio "
                            f"to {target_language} in {elapsed:.3f}s"
                        )

                        return Response(
                            content=wav_data,
                            media_type="audio/wav",
                            headers={
                                "X-Translated-Text": translated_text[:500],
                                "X-Target-Language": target_language,
                                "X-Processing-Time": f"{elapsed:.3f}",
                                "X-Audio-Duration": f"{duration:.3f}",
                                "X-Sample-Rate": str(SAMPLE_RATE),
                            },
                        )
                    else:
                        output_tokens = svc.model.generate(
                            **audio_inputs,
                            tgt_lang=target_language,
                            generate_speech=False,
                        )
                        translated_text = svc.processor.decode(
                            output_tokens[0].tolist()[0], skip_special_tokens=True,
                        )

                        elapsed = time.time() - start_time

                        return {
                            "text": translated_text,
                            "target_language": target_language,
                            "duration_seconds": round(duration, 3),
                            "processing_time": round(elapsed, 3),
                        }

            except Exception as e:
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)

        @api.post("/translate_text")
        async def translate_text(request: Request):
            """Translate text to target language (with optional speech output).

            Accepts JSON:
              - text (str): source text
              - source_language (str): source language code (e.g. "eng")
              - target_language (str): target language code (e.g. "fra")
              - generate_speech (bool, optional): also generate audio (default false)
              - speaker_id (int, optional): speaker voice ID (default 0)
            """
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            text = body.get("text", "")
            source_language = body.get("source_language", "eng")
            target_language = body.get("target_language")
            generate_speech = body.get("generate_speech", False)
            speaker_id = body.get("speaker_id", 0)

            if not text:
                return JSONResponse(status_code=400, content={"error": "text is required"})
            if not target_language:
                return JSONResponse(
                    status_code=400,
                    content={"error": "target_language is required"},
                )

            try:
                import torch

                text_inputs = svc.processor(
                    text=text, src_lang=source_language, return_tensors="pt",
                ).to(svc.device)

                with torch.no_grad():
                    if generate_speech:
                        output = svc.model.generate(
                            **text_inputs,
                            tgt_lang=target_language,
                            speaker_id=speaker_id,
                            generate_speech=True,
                            return_intermediate_token_ids=True,
                        )

                        waveform = output[0].cpu().numpy().squeeze()

                        translated_text = ""
                        if hasattr(output, "sequences") and output.sequences is not None:
                            translated_text = svc.processor.decode(
                                output.sequences[0].tolist(), skip_special_tokens=True,
                            )

                        buf = io.BytesIO()
                        sf.write(buf, waveform, SAMPLE_RATE, format="WAV")
                        buf.seek(0)
                        wav_data = buf.read()

                        elapsed = time.time() - start_time

                        return Response(
                            content=wav_data,
                            media_type="audio/wav",
                            headers={
                                "X-Translated-Text": translated_text[:500],
                                "X-Target-Language": target_language,
                                "X-Processing-Time": f"{elapsed:.3f}",
                            },
                        )
                    else:
                        output_tokens = svc.model.generate(
                            **text_inputs,
                            tgt_lang=target_language,
                            generate_speech=False,
                        )
                        translated_text = svc.processor.decode(
                            output_tokens[0].tolist()[0], skip_special_tokens=True,
                        )

                        elapsed = time.time() - start_time

                        return {
                            "text": translated_text,
                            "source_language": source_language,
                            "target_language": target_language,
                            "processing_time": round(elapsed, 3),
                        }

            except Exception as e:
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

        return api


# --- Local entrypoint --------------------------------------------------------

@app.local_entrypoint()
def main():
    print("SeamlessM4T v2 Translation Service ready")
    print("Deploy with: modal deploy modal/seamless_service.py")
