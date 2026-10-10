from typing import Dict
from .base_env import BaseEnv
from .navigation_env import NavigationEnv

env_aliases: Dict[str, BaseEnv] = {
    "navigation_env": NavigationEnv,
}
