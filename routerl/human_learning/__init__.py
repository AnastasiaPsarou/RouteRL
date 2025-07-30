from .learning_model import (
    GawronModel,
    WeightedModel,
    RandomModel,
    GeneralModel,
    AONModel
)

from .registry import get_learning_model

from .dqn import DQN
from .rmax import Rmax
from .mappo import MAPPO
from .ucb import UCB
from .mbie import MBIE