"""
The third required building block per LeRobot's "Adding a Policy" docs:
a make_<name>_pre_post_processors factory, name-matched to this policy's
registered type string ("act_htp").

DESIGN NOTE: this does NOT reimplement normalization. ACTHTPConfig
inherits ACTConfig's input_features/output_features/normalization_mapping
unchanged, so the generic factory (lerobot.policies.factory.
make_pre_post_processors) should build the identical pipeline already
verified bit-for-bit against the real ACT preprocessor/postprocessor
(state, action round-trip, and image normalization all matched exactly
in prior testing). Delegating here means zero duplicated normalization
logic to keep in sync.

UNVERIFIED ASSUMPTION: that the generic factory function's behavior is
purely config-driven (keys off input_features/output_features/
normalization_mapping/etc.) rather than dispatching internally based on
config.type == "act" specifically. If it turns out to special-case by
type string rather than by feature structure, this thin delegation won't
work and the pipeline construction steps (rename_observations_processor,
to_batch_processor, device_processor, normalizer_processor /
unnormalizer_processor) would need to be built explicitly here instead,
mirroring the exact steps already seen in this checkpoint's own
policy_preprocessor.json / policy_postprocessor.json.
"""

from lerobot.policies.factory import make_pre_post_processors

from .configuration_act_htp import ACTHTPConfig


def make_act_htp_pre_post_processors(
    config: ACTHTPConfig,
    pretrained_path: str | None = None,
    dataset_stats: dict | None = None,
    preprocessor_overrides: dict | None = None,
    postprocessor_overrides: dict | None = None,
):
    return make_pre_post_processors(
        config,
        pretrained_path=pretrained_path,
        dataset_stats=dataset_stats,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )
