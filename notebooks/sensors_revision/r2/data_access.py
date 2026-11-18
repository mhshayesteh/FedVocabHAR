"""Information boundary for new acquisition policies; historical replay stays separate.

This is an API boundary, not a security sandbox against malicious Python code.
Do not pass AuditLabels or full label arrays to a policy.
"""
from dataclasses import dataclass
from collections.abc import Callable
import numpy as np

@dataclass(frozen=True)
class CandidateView:
    window_ids: np.ndarray
    client_ids: np.ndarray
    embeddings: np.ndarray

    def __post_init__(self):
        if len(self.window_ids) != len(self.client_ids) or len(self.window_ids) != len(self.embeddings):
            raise ValueError('Candidate arrays must have equal length')
        if len(set(map(int, self.window_ids))) != len(self.window_ids):
            raise ValueError('Window identifiers must be unique')
        for key in ('window_ids', 'client_ids', 'embeddings'):
            value = np.array(getattr(self, key), copy=True)
            value.flags.writeable = False
            object.__setattr__(self, key, value)

class BudgetedOracle:
    """Every accepted call costs one label, including known/duplicate answers."""
    def __init__(self, lookup: Callable[[int], str], allowed_ids, budget=4):
        if not isinstance(budget, int) or budget < 0: raise ValueError('Budget must be a nonnegative integer')
        self.__lookup = lookup
        self.__allowed = frozenset(map(int, allowed_ids))
        self.__budget = budget
        self.__trace = []

    @property
    def remaining(self): return self.__budget - len(self.__trace)

    @property
    def trace(self): return tuple(dict(row) for row in self.__trace)

    def query(self, window_id):
        window_id = int(window_id)
        if window_id not in self.__allowed: raise ValueError('Window is outside the integration pool')
        if self.remaining == 0: raise RuntimeError('Annotation budget exhausted')
        label = str(self.__lookup(window_id))
        self.__trace.append(dict(call=len(self.__trace)+1, window_id=window_id, label=label))
        return label

@dataclass(frozen=True)
class AuditLabels:
    """Offline diagnostics only. Never pass this object into acquisition code."""
    by_window: dict

    def histogram(self, window_ids):
        labels, counts = np.unique([self.by_window[int(i)] for i in window_ids], return_counts=True)
        return dict(zip(map(str, labels), map(int, counts)))
