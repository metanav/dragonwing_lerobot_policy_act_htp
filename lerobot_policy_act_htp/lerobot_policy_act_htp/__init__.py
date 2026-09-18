"""
Importing this package registers ACTHTPConfig as a PreTrainedConfig
subclass under the type string "act_htp" (via the @register_subclass
decorator in configuration_act_htp.py). This is what makes
`lerobot-rollout --policy.type=act_htp` work once this package is
installed -- LeRobot's config factory looks up registered subclasses by
type string, and installed packages prefixed `lerobot_policy_` are
auto-discovered per LeRobot's "Adding a Policy" documentation.

NOTE: the auto-discovery mechanism itself (exact scanning behavior for
lerobot_policy_-prefixed installed packages) is described in LeRobot's
docs but not independently verified here against your installed
version -- if `--policy.type=act_htp` doesn't show up as a valid choice
after `pip install -e .`, check whether an explicit `import
lerobot_policy_act_htp` somewhere in your launch script is needed as a
fallback to force registration.
"""

from .configuration_act_htp import ACTHTPConfig
from .modeling_act_htp import ACTHTPPolicy
from .processor_act_htp import make_act_htp_pre_post_processors

__all__ = ["ACTHTPConfig", "ACTHTPPolicy", "make_act_htp_pre_post_processors"]
