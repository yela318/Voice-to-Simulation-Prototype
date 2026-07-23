"""
Push-to-talk mic recording -> text, using asr_whisper.py. Enter to start
recording, Enter again to stop.

Usage:
    python voice_capture.py
"""

import numpy as np
import sounddevice as sd

from asr_whisper import transcribe

SAMPLE_RATE = 16000  # Whisper models expect 16kHz mono audio


def record_until_enter() -> np.ndarray:
    input("녹음을 시작하려면 Enter를 누르세요...")
    print("녹음 중... 종료하려면 Enter를 누르세요.")

    frames = []

    def callback(indata, frame_count, time_info, status):
        if status:
            print(status, flush=True)
        frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback
    )
    with stream:
        input()

    if not frames:
        raise RuntimeError("녹음된 오디오가 없습니다. 마이크 입력을 확인하세요.")

    audio = np.concatenate(frames, axis=0).flatten()
    return audio


def main():
    audio = record_until_enter()
    text = transcribe(audio)
    print(f"인식된 텍스트: {text}")


if __name__ == "__main__":
    main()
