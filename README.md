# lerobot_policy_act_htp

Plugs a Qualcomm-AI-Hub-compiled ACT `.tflite` model into `lerobot-rollout`
as a first-class `--policy.type=act_htp`, reusing LeRobot's real CLI,
strategies, robot connection, camera capture, and normalization pipeline
instead of a standalone script.

## Status: unverified on real hardware

Every normalization computation this relies on was verified bit-for-bit
against LeRobot's real pipeline (state, action round-trip, and image
normalization all matched exactly, using `make_pre_post_processors` and
your recorded dataset). What is **not yet verified**:

- Whether `lerobot-rollout` actually discovers this package and accepts
  `--policy.type=act_htp` (auto-discovery mechanism, not independently
  confirmed against your installed LeRobot version).
- Whether subclassing an existing `@register_subclass` config (`ACTConfig`)
  under a new type string works cleanly with LeRobot's config/CLI system.
- Whether `SyncInferenceEngine` calls `.select_action()` with an
  already-preprocessed batch (matches every reference found, but not
  confirmed against your exact installed version's internals).
- Whether the generic `make_pre_post_processors` factory is purely
  feature-driven (works for any config with the right fields) rather than
  dispatching on `config.type == "act"` specifically.
- The compiled model's actual input tensor names (`predict_action_chunk`'s
  name-matching logic assumes "state"/"top"/"wrist" substrings, confirmed
  correct for your specific export in earlier testing, but re-verify if
  you re-export).

Test each of these incrementally rather than assuming the whole chain
works end-to-end on the first run.

## Install

On the IQ-9075 board (needs the QNN/HTP runtime, per earlier setup):

```bash
pip install -e /path/to/lerobot_policy_act_htp[htp]
```

On a dev machine without HTP hardware (e.g. for config testing only):

```bash
pip install -e /path/to/lerobot_policy_act_htp
```

## Prepare your checkpoint directory

Your trained checkpoint's `config.json` says `"type": "act"` -- copy your
checkpoint directory and edit the copy so `lerobot-rollout` routes to this
policy instead of the standard PyTorch `ACTPolicy`:

```bash
cp -r outputs/train/act_so101_test/checkpoints/007000/pretrained_model \
      outputs/train/act_so101_test/checkpoints/007000/pretrained_model_htp

python3 -c "
import json
path = 'outputs/train/act_so101_test/checkpoints/007000/pretrained_model_htp/config.json'
with open(path) as f:
    cfg = json.load(f)
cfg['type'] = 'act_htp'
cfg['tflite_filename'] = 'act_so101_compiled.bin.tflite'
cfg['cache_dir'] = '/home/ubuntu/lerobot_data/tmp'
cfg['model_token'] = 'act_so101_v1'
cfg['htp_skel_library_dir'] = '/usr/lib/rfsa/adsp'
with open(path, 'w') as f:
    json.dump(cfg, f, indent=4)
"

# cache_dir matters a lot, confirmed by real testing: without it,
# sustained control-loop throughput dropped to ~17-18Hz for an entire
# session (not just a one-time startup cost). With a warm cache
# directory reused across runs, a full 60-second session held ~30Hz
# throughout. Point cache_dir at somewhere that persists between runs
# (not /tmp if your board clears it on reboot).

cp act_so101_compiled.bin.tflite \
   outputs/train/act_so101_test/checkpoints/007000/pretrained_model_htp/
```

## Run

```bash
lerobot-rollout \
  --strategy.type=base \
  --policy.type=act_htp \
  --policy.path=outputs/train/act_so101_test/checkpoints/007000/pretrained_model_htp \
  --robot.type=so101_follower \
  --robot.id=my_awesome_follower_arm \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras="{wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
  --task="Grab the yellow cube" \
  --duration=60
```

If this works, you get the full `lerobot-rollout` feature set (dataset
recording strategies, keyboard controls, display_data visualization, safe
teardown) for free, running on the Hexagon NPU -- not just the bare
control loop the standalone script provided.

## Recommended verification order

1. `pip install -e .` (no `[htp]`) on your Mac, confirm
   `python3 -c "import lerobot_policy_act_htp"` succeeds and
   `lerobot-rollout --policy.type=act_htp --help` lists ACTHTPConfig's
   fields (including `tflite_filename`) -- this alone tests config
   registration without touching HTP or the robot at all.
2. On the board, `--dry-run`-equivalent: a very short `--duration=5` run
   with a disconnected/dummy robot config if possible, watching for
   import errors or registration failures before wiring up real hardware.
3. Only then, a real short duration run with the actual robot.
