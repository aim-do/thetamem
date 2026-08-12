"""ΘetaMem: signed multiplicative key lifts for fixed-state sequence memory.

Public surface:

- lift constructors: :func:`branch`, :func:`key`, :func:`key_part`,
  :func:`normalize`, :func:`hadamard`, :func:`outer`, :func:`concat`, and
  :func:`state_lift` for the three canonical state kinds (``"hadamard"``,
  ``"concat"``, ``"outer"``);
- :class:`ThetaMemory` — the memory over projected queries/keys/values;
- :class:`ThetaMemLayer` — a drop-in token mixer around the memory;
- :func:`replay_fit` / :func:`replay_read` — non-causal replay solvers for a
  completed record set;
- :mod:`thetamem.data` — MQAR and MAD task generators for the examples.
"""

from . import data
from .layer import ThetaMemLayer
from .lift import (
    STATE_KINDS,
    Lift,
    branch,
    concat,
    hadamard,
    key,
    key_part,
    normalize,
    outer,
    state_lift,
)
from .memory import BACKENDS, UPDATES, VALUE_CENTERS, ThetaMemory
from .scan import REPLAY_SOLVERS, replay_fit, replay_read

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "branch",
    "key",
    "key_part",
    "normalize",
    "hadamard",
    "outer",
    "concat",
    "state_lift",
    "Lift",
    "STATE_KINDS",
    "UPDATES",
    "BACKENDS",
    "VALUE_CENTERS",
    "REPLAY_SOLVERS",
    "replay_fit",
    "replay_read",
    "ThetaMemory",
    "ThetaMemLayer",
    "data",
]
