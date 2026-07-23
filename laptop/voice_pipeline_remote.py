"""
Records from the mic, transcribes it locally, then SSHes into the
server to run a LIBERO rollout with that text as the prompt.

Requires passwordless SSH access to the server, and a policy server
already running there (scripts/serve_policy.py).

Usage:
  python voice_pipeline_remote.py \
      --args.host <host> \
      --args.user <user> \
      --args.port <port> \
      --args.remote-openpi-path <path-to-openpi-on-the-server> \
      --args.task-id 0

Optional: --args.asr-backend naver (default is whisper),
--args.language ko (default is en -- whisper auto-detects either way,
naver needs to be told and chains through translate_naver.py if ko),
--args.identity-file <key> if ssh needs a specific key,
--args.remote-home <path> for accounts with a LIBERO config quirk,
--args.audio-file <file> to transcribe an existing file instead of
recording live.
"""

import dataclasses
import shlex
import subprocess

import tyro

from voice_capture import SAMPLE_RATE, record_until_enter


@dataclasses.dataclass
class Args:
    host: str = tyro.MISSING
    user: str = tyro.MISSING
    remote_openpi_path: str = tyro.MISSING

    asr_backend: str = "whisper"  # "whisper" (asr_whisper.py) or "naver" (asr_naver.py)
    language: str = "en"  # "en" or "ko" -- the language spoken into the mic

    port: int = 22  # SSH port, if not the default
    identity_file: str = ""  # path to private key, if ssh needs one specified

    remote_home: str = ""  # override HOME for the remote command, if needed
    audio_file: str = ""  # transcribe this file instead of recording live

    task_id: int = 0
    task_suite_name: str = "libero_spatial"


def main(args: Args) -> None:
    if args.asr_backend == "whisper":
        from asr_whisper import transcribe
    elif args.asr_backend == "naver":
        from asr_naver import transcribe
    else:
        raise ValueError(f"Unknown asr_backend: {args.asr_backend!r} (expected 'whisper' or 'naver')")

    if args.audio_file:
        print(f"Using existing audio file: {args.audio_file}")
        audio_input = args.audio_file
    else:
        audio = record_until_enter()
        audio_input = audio if args.asr_backend == "whisper" else (audio, SAMPLE_RATE)

    if args.asr_backend == "whisper":
        # Whisper auto-detects the spoken language and translates to
        # English by default (see asr_whisper.py) -- passing the language
        # explicitly just skips auto-detection, it's not required.
        text = transcribe(audio_input, language=args.language)
    else:
        naver_lang = "Kor" if args.language == "ko" else "Eng"
        text = transcribe(audio_input, lang=naver_lang)
        if args.language == "ko":
            from translate_naver import translate

            print(f'Transcribed (Korean): "{text}"')
            text = translate(text, source="ko", target="en")

    print(f'Transcribed: "{text}"')

    home_override = ""
    if args.remote_home:
        home_override = f"mkdir -p {shlex.quote(args.remote_home)} && export HOME={shlex.quote(args.remote_home)} && "

    remote_command = (
        f"{home_override}"
        f"cd {shlex.quote(args.remote_openpi_path)} && "
        f"source examples/libero/.venv/bin/activate && "
        f"export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero && "
        f"MUJOCO_GL=egl python examples/libero/voice_demo.py "
        f"--args.prompt {shlex.quote(text)} "
        f"--args.task-id {args.task_id} "
        f"--args.task-suite-name {shlex.quote(args.task_suite_name)}"
    )

    ssh_cmd = ["ssh", "-p", str(args.port)]
    if args.identity_file:
        ssh_cmd += ["-i", args.identity_file]
    ssh_cmd += [f"{args.user}@{args.host}", remote_command]

    print(f"Running on {args.host}:{args.port}...")
    subprocess.run(ssh_cmd, check=True)


if __name__ == "__main__":
    tyro.cli(main)