# lerobot_policy_act_htp

Plugs a Qualcomm AI Hub compiled/exported ACT `.tflite` model into `lerobot-rollout`
as a first-class `--policy.type=act_htp`, reusing LeRobot's real CLI,
strategies, robot connection, camera capture, and normalization pipeline.

## Install

On the IQ-9075 board (needs the QNN/HTP runtime installed):

```bash
pip install -e /path/to/lerobot_policy_act_htp
```

## Prepare your checkpoint directory

Modify the trained checkpoint's `config.json` says `"type": "act"` to `"type": "act_htp"`
so `lerobot-rollout` routes to this policy instead of the standard PyTorch `ACTPolicy`

```bash
cp act_so101_compiled.bin.tflite \
   outputs/train/act_so101_test/checkpoints/007000/pretrained_model_htp/
```

## Run

```bash
lerobot-rollout \
  --strategy.type=base \
  --policy.path=outputs/train/act_so101_test/checkpoints/last/pretrained_model_htp \
  --robot.type=so101_follower \
  --robot.id=my_awesome_follower_arm \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras="{wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
  --task="Grab the yellow cube" \
  --duration=60
```
