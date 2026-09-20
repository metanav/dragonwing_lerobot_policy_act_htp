from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("act_htp")
@dataclass
class ACTHTPConfig(ACTConfig):
    type: str = "act_htp"

    # Filename of the Qualcomm AI Hub compiled model
    tflite_filename: str = "act_so101_compiled.bin.tflite"

    htp_performance_mode: str = "2"

    # QNN context caching
    cache_dir: str | None = None
    
    model_token: str = "act_so101_v1"

    prefetch_threshold: int = 13

    # library path
    htp_skel_library_dir: str = "/usr/lib/rfsa/adsp"
