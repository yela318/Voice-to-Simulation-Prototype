"""
Speech-to-text using Naver CLOVA Speech Recognition (CSR), as an
alternative to asr_whisper.py. Same interface: give it audio, get back text.
Built for short (<=60s, <=3MB) command-style audio.

Setup:
  export NCP_CLIENT_ID=<Client ID>
  export NCP_CLIENT_SECRET=<Client Secret>
(keys are read from the environment, never hardcoded here)

Docs: https://api.ncloud-docs.com/docs/ai-naver-clovaspeechrecognition-stt
"""

import io
import os
import wave

import numpy as np
import requests

API_URL = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt"


def _numpy_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 [-1, 1] mono samples (e.g. from voice_capture.py's
    sounddevice recording) into 16-bit PCM WAV bytes, since CSR expects
    an actual audio file/binary, not a raw array."""
    clipped = np.clip(samples, -1.0, 1.0)
    int16_samples = (clipped * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(int16_samples.tobytes())
    return buf.getvalue()


def transcribe(audio_input, lang: str = "Eng") -> str:
    """
    audio_input: str file path (wav/mp3/aac/ac3/ogg/flac), OR
                 (samples, sample_rate) tuple of a float32 mono numpy array
                 -- e.g. straight from voice_capture.py's record_until_enter()
    lang: "Kor" | "Eng" | "Jpn" | "Chn"
    """
    client_id = os.environ["NCP_CLIENT_ID"]
    client_secret = os.environ["NCP_CLIENT_SECRET"]

    if isinstance(audio_input, str):
        with open(audio_input, "rb") as f:
            audio_bytes = f.read()
    else:
        samples, sample_rate = audio_input
        audio_bytes = _numpy_to_wav_bytes(samples, sample_rate)

    response = requests.post(
        API_URL,
        params={"lang": lang},
        headers={
            "x-ncp-apigw-api-key-id": client_id,
            "x-ncp-apigw-api-key": client_secret,
            "Content-Type": "application/octet-stream",
        },
        data=audio_bytes,
        timeout=15,
    )
    response.raise_for_status()
    return response.json()["text"]
