"""The full physical-policy catalog, including the two new study groups.

The original zoo.PHYSICAL list remains its frozen 17-policy reproduction set.
Use these names for the expanded catalog and the linked study runners to
compare newly added controllers under matched protocols.
"""

import zoo
from targeted_policies import POLICIES as GAUSSIAN_CONTROLLERS
from broad_policies import POLICIES as BROAD_CONTROLLERS


NEW_PHYSICAL_RULES = {**GAUSSIAN_CONTROLLERS, **BROAD_CONTROLLERS}
ALL_PHYSICAL_NAMES = tuple(zoo.PHYSICAL) + tuple(NEW_PHYSICAL_RULES)

assert len(NEW_PHYSICAL_RULES) == 13
assert len(ALL_PHYSICAL_NAMES) == len(set(ALL_PHYSICAL_NAMES)) == 30


def collect_new(config, name, episodes, seed):
    """Collect one new controller's physical observation/action pairs."""
    controller = NEW_PHYSICAL_RULES[name]()
    return zoo.collect_physical(config, controller, episodes, seed)
