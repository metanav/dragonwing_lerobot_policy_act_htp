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
