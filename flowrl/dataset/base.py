from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import collections
import flax

Batch = collections.namedtuple(
    'Batch',
    ['obs', 'action', 'reward', 'terminal', 'next_obs']
)