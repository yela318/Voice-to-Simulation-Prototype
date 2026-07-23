"""
Audio file -> ASR -> LIBERO rollout, in one step (no SSH involved --
this runs entirely on the server, using an audio file already there).
ASR backend is selectable with --args.asr-backend.

Usage (inside the LIBERO venv, same as voice_demo.py):
  MUJOCO_GL=egl python examples/libero/voice_to_sim.py \
      --args.audio-path examples/libero/voice.m4a \
      --args.task-id 0 \
      --args.asr-backend whisper   # or: naver

Requires a policy server (e.g. pi05_libero) running in a separate
session, same as voice_demo.py. Naver backend also needs
NCP_CLIENT_ID / NCP_CLIENT_SECRET set (see asr_naver.py).
"""

import dataclasses
import logging

import tyro

from voice_demo import Args as SimArgs
from voice_demo import run_once


@dataclasses.dataclass
class Args:
    audio_path: str = tyro.MISSING  # voice command audio file to transcribe and run
    asr_backend: str = "whisper"  # "whisper" (local, asr_whisper.py) or "naver" (asr_naver.py)

    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = "libero_spatial"
    task_id: int = 0

    num_steps_wait: int = 10
    max_steps: int = 220
    seed: int = 7

    video_out_path: str = "data/libero/voice_demo"


def main(args: Args) -> None:
    if args.asr_backend == "whisper":
        from asr_whisper import transcribe
    elif args.asr_backend == "naver":
        from asr_naver import transcribe
    else:
        raise ValueError(f"Unknown asr_backend: {args.asr_backend!r} (expected 'whisper' or 'naver')")

    text = transcribe(args.audio_path)
    logging.info(f"[ASR:{args.asr_backend}] '{args.audio_path}' -> \"{text}\"")

    sim_args = SimArgs(
        host=args.host,
        port=args.port,
        resize_size=args.resize_size,
        replan_steps=args.replan_steps,
        task_suite_name=args.task_suite_name,
        task_id=args.task_id,
        prompt=text,  # <- the actual voice-to-sim connection point
        num_steps_wait=args.num_steps_wait,
        max_steps=args.max_steps,
        seed=args.seed,
        video_out_path=args.video_out_path,
    )
    run_once(sim_args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
