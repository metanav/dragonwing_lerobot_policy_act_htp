"""
Config for the HTP-deployed ACT policy.

DESIGN NOTE: this subclasses ACTConfig rather than building a config from
scratch. Since ACT's chunk_size, n_action_steps, input_features,
output_features, and normalization_mapping are UNCHANGED by moving
inference to a compiled .tflite graph -- only the compute backend
changes -- inheriting means the generic `make_pre_post_processors`
factory (see processor_act_htp.py) can build the exact same
normalize/unnormalize pipeline already verified bit-for-bit against the
real ACT pipeline, with zero reimplementation.

UNVERIFIED ASSUMPTION: that subclassing an existing @register_subclass
dataclass config (ACTConfig) and re-registering the subclass under a new
type string works cleanly with LeRobot's draccus-based config/CLI
system. This is a reasonable design given dataclass inheritance is
generally supported, but has not been tested against your installed
LeRobot version. If registration or CLI parsing breaks on this, the
fallback is to build ACTHTPConfig as a plain PreTrainedConfig subclass
that duplicates ACTConfig's fields explicitly instead of inheriting.
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("act_htp")
@dataclass
class ACTHTPConfig(ACTConfig):
    type: str = "act_htp"

    # Filename of the Qualcomm-AI-Hub-compiled model, expected to sit
    # alongside config.json in the same pretrained_path directory that
    # lerobot-rollout --policy.path points to.
    tflite_filename: str = "act_so101_compiled.bin.tflite"

    # QNN HTP delegate performance mode. Verify against your QNN SDK
    # version's documentation what each numeric value maps to (burst /
    # sustained-high-performance / balanced / low-power, etc.) -- "2" was
    # carried over from earlier testing without confirming its exact
    # meaning.
    htp_performance_mode: str = "2"

    # QNN context caching. Confirmed via real on-device testing to matter
    # a great deal, not just for first-load time: without a populated
    # cache, sustained control-loop throughput dropped to ~17-18Hz
    # (vs. the ~30Hz target) for an entire session, not just during the
    # initial graph-preparation stage -- suggesting the backend may
    # re-validate or re-establish session state on every invoke() call
    # without a cache, not only once at startup. With a warm cache
    # (matching cache_dir + model_token to a previous run), sustained
    # throughput held at ~30Hz for a full 60s session. Leave cache_dir
    # unset to disable caching (reverts to original slower behavior);
    # set it to a writable directory that persists across runs.
    cache_dir: str | None = None
    model_token: str = "act_so101_v1"

    # Start computing the next chunk once this many actions remain in
    # the queue, rather than waiting for it to hit zero. At 30fps,
    # threshold=13 leaves ~433ms of runway. If you see "prefetch did not
    # finish in time" warnings in the logs, raise this value -- it means
    # actual on-device inference latency (possibly including contention
    # with camera capture or other board load) exceeds the runway this
    # provides.
    prefetch_threshold: int = 13

    #lib
    htp_skel_library_dir: str = "/usr/lib/rfsa/adsp"
