"""C1 Aggregate-Centroid-Sensor-v1: independently written qualification core.

Paper-inspired selector only, not a native FederatedTypiClust reproduction.
Clients are simulated in process. No semantic labels, model training or decoder.
The coordinator consumes aggregate statistics and candidate ID/distance replies.
"""
from dataclasses import dataclass
from numbers import Integral
import numpy as np

PROFILE = 'Aggregate-Centroid-Sensor-v1'


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(name)
    return int(value)


@dataclass(frozen=True)
class Candidate:
    client: int
    window: int
    cluster: int
    squared_distance: float


class Client:
    """Local rows remain inside this simulation object; this is not RPC/security."""
    def __init__(self, client_id, window_ids, features):
        self.client_id = integer(client_id, 'client ID')
        self.ids = np.array([integer(w, 'window ID') for w in window_ids], dtype=np.int64)
        self._z = np.array(features, dtype=np.float64, copy=True)
        if self._z.ndim != 2 or self._z.shape[1] == 0 or len(self.ids) != len(self._z):
            raise ValueError('feature/ID shape')
        if len(set(self.ids.tolist())) != len(self.ids) or not np.isfinite(self._z).all():
            raise ValueError('duplicate IDs or nonfinite features')
        if len(self.ids) and np.any(np.linalg.norm(self._z, axis=1) == 0):
            raise ValueError('zero embedding')
        order = np.argsort(self.ids, kind='stable')
        self.ids, self._z = self.ids[order], self._z[order]
        self.dimension = self._z.shape[1]
        self._assignments = None

    def initial_statistics(self):
        return len(self.ids), self._z.sum(axis=0), float(np.einsum('ij,ij->', self._z, self._z))

    def assign_reduce(self, centroids):
        distances = np.sum((self._z[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        self._assignments = np.argmin(distances, axis=1)
        counts = np.bincount(self._assignments, minlength=len(centroids)).astype(np.int64)
        sums = np.zeros_like(centroids)
        np.add.at(sums, self._assignments, self._z)
        return counts, sums

    def propose(self, cluster, centroid, used):
        eligible = np.flatnonzero((self._assignments == cluster) & ~np.isin(self.ids, list(used)))
        if not len(eligible):
            return None
        distances = np.sum((self._z[eligible] - centroid) ** 2, axis=1)
        index = min(range(len(eligible)), key=lambda j: (float(distances[j]), int(self.ids[eligible[j]])))
        pos = eligible[index]
        return Candidate(self.client_id, int(self.ids[pos]), cluster, float(distances[index]))


def canonical_clients(clients):
    clients = sorted(clients, key=lambda c: c.client_id)
    if not clients or len({c.client_id for c in clients}) != len(clients):
        raise ValueError('nonempty unique client collection required')
    if len({c.dimension for c in clients}) != 1:
        raise ValueError('embedding dimensions differ')
    all_ids = [int(w) for c in clients for w in c.ids]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError('window IDs must be globally unique')
    return clients


def initialize(clients, k, seed=20261117):
    """Own profile choice: aggregate mean/variance plus seeded directions.

    No individual target embedding is used as a server initialization anchor.
    Input embeddings are used unchanged; centroids are arithmetic means.
    """
    k = integer(k, 'cluster count', 1)
    seed = integer(seed, 'seed')
    statistics = [c.initial_statistics() for c in clients]
    total = sum(row[0] for row in statistics)
    if not total:
        raise ValueError('no rows for initialization')
    mean = sum((row[1] for row in statistics), np.zeros(clients[0].dimension)) / total
    variance = max(0., sum(row[2] for row in statistics) / total - float(mean @ mean))
    directions = np.random.default_rng(seed).normal(size=(k, len(mean)))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions /= np.maximum(norms, np.finfo(float).tiny)
    return mean + np.sqrt(variance) * directions


def fit(clients, initial_centroids, max_iter=100, tolerance=1e-8):
    max_iter = integer(max_iter, 'iteration cap', 1)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('tolerance')
    centers = np.array(initial_centroids, dtype=np.float64, copy=True)
    if centers.ndim != 2 or centers.shape[1] != clients[0].dimension or not len(centers) or not np.isfinite(centers).all():
        raise ValueError('initial centroids')
    converged = False
    for iteration in range(1, max_iter + 1):
        replies = [c.assign_reduce(centers) for c in clients]
        counts = sum((r[0] for r in replies), np.zeros(len(centers), dtype=np.int64))
        sums = sum((r[1] for r in replies), np.zeros_like(centers))
        updated = centers.copy()
        occupied = counts > 0
        updated[occupied] = sums[occupied] / counts[occupied, None]
        shift = float(np.max(np.linalg.norm(updated - centers, axis=1)))
        centers = updated
        if shift <= tolerance:
            converged = True
            break
    # Final local assignment must correspond to the returned centers, including
    # when the iteration cap was reached. Empty centers remain unchanged.
    replies = [c.assign_reduce(centers) for c in clients]
    counts = sum((r[0] for r in replies), np.zeros(len(centers), dtype=np.int64))
    return centers, counts, dict(iterations=iteration, converged=converged,
                                termination='tolerance' if converged else 'iteration_cap')


def select(clients, budget=4, *, client_caps=None, cap_mode='global',
           initial_centroids=None, seed=20261117, max_iter=100, tolerance=1e-8):
    """Return a pre-oracle query plan. No labels/answers are accepted.

    global: only global B binds; each client may spend up to B. This relaxes
      the paper's fixed client allocation and is named explicitly.
    allocated: caller supplies nonnegative caps summing exactly to global B.
      These must be matched for every arm in any scientific comparison.
    """
    budget = integer(budget, 'global budget')
    if budget > 4:
        raise ValueError('qualification profile supports at most four calls')
    clients = canonical_clients(clients)
    if cap_mode == 'global':
        if client_caps is not None:
            raise ValueError('global mode cannot also specify local caps')
        caps = {c.client_id: budget for c in clients}
    elif cap_mode == 'allocated':
        if client_caps is None or set(client_caps) != {c.client_id for c in clients}:
            raise ValueError('allocated caps must name every client')
        caps = {key: integer(value, 'client cap') for key, value in client_caps.items()}
        if sum(caps.values()) != budget:
            raise ValueError('allocated caps must sum to global budget')
    else:
        raise ValueError('unknown cap mode')
    total = sum(len(c.ids) for c in clients)
    spent = {c.client_id: 0 for c in clients}
    result = dict(profile=PROFILE, cap_mode=cap_mode, budget=budget, client_caps=caps,
                  plan=[], spent_by_client=spent, unused_budget=budget,
                  semantic_answers_read=0, initial_target_labels=0,
                  fit=None, centroids=[], cluster_counts=[])
    if not total or not budget:
        result['stop_reason'] = 'empty_pool' if not total else 'zero_budget'
        return result
    if initial_centroids is None:
        initial_centroids = initialize(clients, min(budget, total), seed)
    centers, counts, fit_status = fit(clients, initial_centroids, max_iter, tolerance)
    used = set()
    paid_counts = np.zeros(len(centers), dtype=np.int64)
    for call in range(budget):
        # The geometry stays fixed during the batch. All calls, including later
        # known/duplicate semantic answers, count equally toward paid counts.
        proposals = {}
        for cluster in range(len(centers)):
            local = [c.propose(cluster, centers[cluster], used)
                     for c in clients if spent[c.client_id] < caps[c.client_id]]
            local = [candidate for candidate in local if candidate is not None]
            if local:
                proposals[cluster] = min(local, key=lambda p: (p.squared_distance, p.client, p.window))
        if not proposals:
            break
        cluster = min(proposals, key=lambda j: (int(paid_counts[j]), -int(counts[j]), j))
        candidate = proposals[cluster]
        result['plan'].append(dict(call=call + 1, client_id=candidate.client,
                                   window_id=candidate.window, cluster=cluster,
                                   squared_distance=candidate.squared_distance,
                                   prior_paid_in_cluster=int(paid_counts[cluster]),
                                   cluster_population=int(counts[cluster])))
        used.add(candidate.window)
        spent[candidate.client] += 1
        paid_counts[cluster] += 1
    result.update(unused_budget=budget - len(used), centroids=centers.tolist(),
                  cluster_counts=counts.tolist(), fit=fit_status,
                  stop_reason='budget_reached' if len(used) == budget else 'feasible_pool_exhausted')
    return result
