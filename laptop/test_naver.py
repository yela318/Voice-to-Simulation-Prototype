"""
Checks that the Naver CSR backend is set up correctly.
Requires NCP_CLIENT_ID / NCP_CLIENT_SECRET to be set in the environment.

Usage:
    python test_naver.py <path_to_audio_file>
"""

import sys

from asr_naver import transcribe


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_naver.py <path_to_audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]
    text = transcribe(audio_path, lang="Eng")
    print(text)


if __name__ == "__main__":
    main()
