# Voice-to-Simulation-Prototype

Voice → text → LIBERO robot simulation, using faster-whisper or Naver
CLOVA Speech Recognition (CSR) for speech-to-text, and a pi0/pi0.5
policy (openpi) to turn that text into robot actions.

## Architecture

Recording and speech-to-text happen on a **laptop** (wherever the
microphone is). The **server** only ever receives already-transcribed
text and runs the LIBERO rollout — it does not need a microphone or
any audio hardware.

```
[laptop]  mic → record → transcribe (faster-whisper / Naver CSR)
              → SSH → text as --args.prompt
                                        ↓
[server]                        LIBERO rollout (voice_demo.py)
```

## Files

### Laptop (local)

| File | Role |
|---|---|
| `voice_capture.py` | Push-to-talk mic recording |
| `asr_whisper.py` | ASR backend: local [faster-whisper](https://github.com/SYSTRAN/faster-whisper) |
| `asr_naver.py` | ASR backend: [Naver CLOVA Speech Recognition](https://api.ncloud-docs.com/docs/ai-naver-clovaspeechrecognition) (cloud API) |
| `voice_pipeline_remote.py` | **Main entry point.** Record → transcribe → SSH into the server → run a LIBERO rollout there with the transcribed text as the prompt |
| `transcribe_file.py` | Install check: confirms faster-whisper works locally (no mic needed) |
| `test_naver.py` | Install check: confirms Naver API credentials work (no mic needed) |
| `voice.m4a` | Sample audio for the install-check scripts above |
| `requirements.txt` | Dependencies for all of the above |

### Server (`examples/libero/` inside your openpi checkout)

| File | Role |
|---|---|
| `voice_demo.py` | **Core.** Single-episode LIBERO rollout using a CLI-supplied prompt |
| `voice_to_sim.py` | Standalone test: audio file (already on the server) → LIBERO rollout, without going through SSH. Useful for isolating "is the sim side broken" from "is the SSH pipeline broken" |
| `asr_whisper.py` / `asr_naver.py` | Needed by `voice_to_sim.py` above |
| `voice.m4a` | Sample audio for the `voice_to_sim.py` test above |
| `requirements-voice.txt` | This repo's `requirements.txt`, copied under a different name so it doesn't collide with openpi's own `examples/libero/requirements.txt` |

## Setup — Server

**1. Get openpi:**
```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

**2. Get this repo's files onto the server, inside `examples/libero/`:**
```bash
git clone --depth 1 https://github.com/<your-username>/Voice-to-Simulation-Prototype.git /tmp/vtl \
    && cp /tmp/vtl/server/*.py /tmp/vtl/server/*.m4a examples/libero/ \
    && cp /tmp/vtl/server/requirements.txt examples/libero/requirements-voice.txt \
    && rm -rf /tmp/vtl
```

**3. Set up the LIBERO environment.** You'll need at least two terminal
panes (e.g. [byobu](https://byobu.org/)):

```bash
# Terminal pane #1:
cd <path-to-your-openpi>
git submodule update --init --recursive
```

```bash
# Terminal pane #2:
cd <path-to-your-openpi>
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
uv pip install -r examples/libero/requirements-voice.txt
```

> `requirements-voice.txt` should contain `tokenizers<0.21` pinned
> explicitly — without it, `uv pip install` may try to build the
> newest `tokenizers` from source and fail on Python 3.8 (no prebuilt
> wheel exists for it on that version). Check the file before
> continuing if this hasn't been confirmed yet.

**4. Test the server side, standalone (no SSH, no laptop needed yet):**
```bash
# Terminal pane #1:
uv run scripts/serve_policy.py --env LIBERO
```
```bash
# Terminal pane #2 (venv still active):
MUJOCO_GL=egl python examples/libero/voice_to_sim.py \
    --args.audio-path examples/libero/voice.m4a \
    --args.task-id 0
```
If this produces a rollout video without errors, the server side is
confirmed working, independent of the laptop/SSH pipeline.

## Setup — Laptop

**1. Get this repo and go into the laptop folder:**
```bash
git clone https://github.com/<your-username>/Voice-to-Simulation-Prototype.git
cd Voice-to-Simulation-Prototype/laptop
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Test the ASR install checks (no mic needed):**
```bash
python transcribe_file.py voice.m4a
```
```bash
# only if using the Naver backend:
export NCP_CLIENT_ID=<your Client ID>
export NCP_CLIENT_SECRET=<your Client Secret>
python test_naver.py voice.m4a
```
(Get Naver credentials by registering an Application under **AI·NAVER
API** in the [Naver Cloud Platform console](https://www.ncloud.com/) —
these are different from NCP's general platform Access Key / Secret
Key.)

**4. Confirm plain SSH access to the server works** (outside of any
script, using whatever host/port/user/key you normally use):
```bash
ssh -p <port> <user>@<host>
```

**5. Test the full remote pipeline, without talking into the mic yet**
(reuses `voice.m4a` instead of live recording, to isolate SSH/server
issues from ASR issues):
```bash
python voice_pipeline_remote.py \
    --args.host <host> \
    --args.user <user> \
    --args.port <port> \
    --args.remote-openpi-path <path-to-openpi-on-the-server> \
    --args.audio-file voice.m4a \
    --args.task-id 0
```
Add `--args.identity-file <path-to-key>` if SSH doesn't already know
which key to use for this host. Add `--args.remote-home <path>` only
if the server's LIBERO config needs a HOME override for this account
(a shared-account quirk — most setups won't need this). Swap
`--args.audio-file` to point at any other audio file to test different
prompts without needing to re-record each time. Add `--args.asr-backend
naver` to use Naver CSR instead of the default (faster-whisper).

**6. Once step 5 works, drop `--args.audio-file` to use the live
mic instead:**
```bash
python voice_pipeline_remote.py \
    --args.host <host> \
    --args.user <user> \
    --args.port <port> \
    --args.remote-openpi-path <path-to-openpi-on-the-server> \
    --args.task-id 0
```

## System dependencies

If `voice_pipeline_remote.py` fails on the laptop with `OSError:
PortAudio library not found`:
- Linux: `sudo apt-get install -y libportaudio2`
- Mac: `brew install portaudio`

## Attribution

- `voice_demo.py` is a modified version of openpi's
  [`examples/libero/main.py`](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py)
  (Apache License 2.0, Physical Intelligence). Changes: reduced to a
  single episode per run, and uses a user-supplied prompt instead of
  the task's default `task.language`. See
  [LICENSE-openpi](LICENSE-openpi) for the original license text.
- `voice.m4a` is a short sample recording used to test the ASR step
  without needing a live microphone. The file contains the sentence:
  "Pick up the black bowl between the plate and place it on the plate."

## License

Portions derived from openpi remain subject to Apache License 2.0
(see [LICENSE-openpi](LICENSE-openpi)).
