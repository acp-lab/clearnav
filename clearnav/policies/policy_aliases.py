from typing import Dict
from .mlp_policy import MlpPolicy
from .multi_input_policy import MultiInputPolicy

policy_aliases: Dict[str, MlpPolicy] = {
    "MlpPolicy": MlpPolicy,
    "MultiInputPolicy": MultiInputPolicy,
}
