"""
Runs one LIBERO episode using a given language prompt, instead of the
task suite's default instruction. Based on openpi's examples/libero/main.py.

Usage (inside the LIBERO venv):
  MUJOCO_GL=egl python examples/libero/voice_demo.py \
      --args.prompt "pick up the black bowl" \
      --args.task-id 0

Requires a policy server (e.g. pi05_libero) running in a separate session.
"""

import dataclasses
import logging
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = "libero_spatial"
    task_id: int = 0  # Reuse this task's scene only (object layout, etc.)
    prompt: str = tyro.MISSING  # Overrides task.language as the language instruction

    num_steps_wait: int = 10
    max_steps: int = 220
    seed: int = 7

    video_out_path: str = "data/libero/voice_demo"


def run_once(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)

    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(args.seed)

    original_prompt = task.language
    logging.info(f"Default prompt for this scene: '{original_prompt}'")
    logging.info(f"Prompt actually used: '{args.prompt}'")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    env.reset()
    obs = env.set_init_state(initial_states[0])

    replay_images = []
    action_plan = []
    t = 0
    done = False

    while t < args.max_steps + args.num_steps_wait:
        if t < args.num_steps_wait:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
        )
        replay_images.append(img)

        if not action_plan:
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": np.concatenate(
                    (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ),
                "prompt": args.prompt,  # The only real change from the original script
            }
            action_chunk = client.infer(element)["actions"]
            action_plan = list(action_chunk[: args.replan_steps])

        action = action_plan.pop(0)
        obs, reward, done, info = env.step(action.tolist())
        if done:
            break
        t += 1

    suffix = "success" if done else "incomplete"
    prompt_segment = args.prompt.replace(" ", "_")[:40]
    out_path = pathlib.Path(args.video_out_path) / f"voice_demo_{prompt_segment}_{suffix}.mp4"
    imageio.mimwrite(out_path, [np.asarray(x) for x in replay_images], fps=10)
    logging.info(f"Done (done={done}). Video saved to: {out_path}")


def _quat2axisangle(quat):
    import math

    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(run_once)
