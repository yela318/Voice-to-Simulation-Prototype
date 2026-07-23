"""
Local speech-to-text using faster-whisper. Used by voice_to_sim.py.
"""

from typing import Optional
from faster_whisper import WhisperModel

_model: Optional[WhisperModel] = None  # loaded once, kept in memory

# Defaults tuned for short, clear robot-command style utterances.
DEFAULT_MODEL_SIZE = "small"
DEFAULT_DEVICE = "cpu"          # switch to "cuda" if a GPU is available
DEFAULT_COMPUTE_TYPE = "int8"   # use "float16" if DEFAULT_DEVICE == "cuda"


def load_model(
    model_size: str = DEFAULT_MODEL_SIZE,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
) -> WhisperModel:
    """Load (once) and return the shared WhisperModel instance."""
    global _model
    if _model is None:
        _model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _model


def transcribe(
    audio_input,
    language: Optional[str] = None,
    task: str = "translate",
    initial_prompt: Optional[str] = None,
) -> str:
    """
    Transcribe an audio file path (str) or a raw waveform
    (numpy float32 array, 16kHz mono) into English text.

    audio_input: str file path, or np.ndarray waveform
    language: None (default) lets Whisper auto-detect the spoken
              language from the first ~30s of audio. Pass "en"/"ko"/etc.
              to force a specific one (slightly faster, avoids rare
              misdetection).
    task: "translate" (default) always outputs English, regardless of
          the spoken language -- for English audio this is the same
          as "transcribe". Use "transcribe" to keep the original
          language's text instead of translating it.
    initial_prompt: optional domain-vocabulary hint (e.g. task object names)
                    to bias decoding toward expected LIBERO-style phrasing
    """
    model = load_model()
    segments, info = model.transcribe(
        audio_input,
        language=language,
        task=task,
        vad_filter=True,                 # trims leading/trailing silence, reduces hallucination
        condition_on_previous_text=False,  # each command is an independent utterance
        initial_prompt=initial_prompt,
    )
    text = " ".join(seg.text.strip() for seg in segments)
    return text.strip()
