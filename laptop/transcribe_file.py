"""
Checks that the local faster-whisper backend is set up correctly.

Usage:
    python transcribe_file.py <path_to_audio_file>
"""

import sys

from asr_whisper import transcribe


def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe_file.py <path_to_audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]
    text = transcribe(audio_path)
    print(text)


if __name__ == "__main__":
    main()
