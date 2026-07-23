"""
Text translation using Naver Papago NMT. Meant to be chained after
asr_naver.py's transcribe() when the source audio isn't in English --
CSR only transcribes in the language you specify, it doesn't translate,
so this is a separate step (unlike Whisper, which can transcribe and
translate to English in one call).

Papago Translation and CLOVA Speech Recognition are registered as
separate Applications in the Naver Cloud Platform console, each with
their own Client ID / Client Secret -- these are NOT the same
credentials as asr_naver.py's NCP_CLIENT_ID / NCP_CLIENT_SECRET.

Setup:
  export NCP_PAPAGO_CLIENT_ID=<Client ID>
  export NCP_PAPAGO_CLIENT_SECRET=<Client Secret>

Docs: https://api.ncloud-docs.com/docs/ai-naver-papagonmt
"""

import os

import requests

API_URL = "https://papago.apigw.ntruss.com/nmt/v1/translation"


def translate(text: str, source: str = "ko", target: str = "en") -> str:
    """
    source / target: language codes, e.g. "ko", "en", "ja", "zh-CN"
    """
    client_id = os.environ["NCP_PAPAGO_CLIENT_ID"]
    client_secret = os.environ["NCP_PAPAGO_CLIENT_SECRET"]

    response = requests.post(
        API_URL,
        headers={
            "X-NCP-APIGW-API-KEY-ID": client_id,
            "X-NCP-APIGW-API-KEY": client_secret,
        },
        data={"source": source, "target": target, "text": text},
        timeout=15,
    )
    response.raise_for_status()
    return response.json()["message"]["result"]["translatedText"]
