"""
Checks that the Naver Papago translation backend is set up correctly.
Requires NCP_PAPAGO_CLIENT_ID / NCP_PAPAGO_CLIENT_SECRET to be set in
the environment -- a separate Application/credentials from CSR's
NCP_CLIENT_ID / NCP_CLIENT_SECRET.

Usage:
    python test_translate.py "안녕하세요"
"""

import sys

from translate_naver import translate


def main():
    if len(sys.argv) < 2:
        print('Usage: python test_translate.py "text to translate"')
        sys.exit(1)

    text = sys.argv[1]
    translated = translate(text, source="ko", target="en")
    print(translated)


if __name__ == "__main__":
    main()
