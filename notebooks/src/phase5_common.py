"""Shared Phase 5 machinery for FedVocabHAR (identical logic to FedVocabHAR_Phase5A_Baselines.ipynb).

Import from a Colab notebook with:
    import sys; sys.path.insert(0, '/content/drive/MyDrive/FedVocabHAR/colab/src'); import phase5_common as P5
"""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import json, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import precision_recall_fscore_support
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Config:
    project: str = '/content/drive/MyDrive/FedVocabHAR'
    seed: int = 42
    probe_per_subject_activity: int = 220
    max_candidates_per_client: int = 600
    min_local_cluster_size: int = 12
    min_agreeing_clients: int = 2
    label_query_budget: int = 4
    novelty_false_alarm_quantile: float = .02
    local_distance_quantile: float = .90
    agreement_similarity_quantile: float = .10
    random_query_seeds: int = 20
    gmm_max_components: int = 8
    bootstrap_resamples: int = 10000


CFG = Config()
PROJECT = Path(CFG.project)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def assert_locked(cfg=None):
    cfg = cfg or CFG
    lock = json.loads((PROJECT / 'artifacts' / 'phase3d' / 'phase3d_policy_lock.json').read_text())
    assert lock['selected_policy'] == 'ranked_mnn'
    for k in ('max_candidates_per_client', 'min_local_cluster_size', 'min_agreeing_clients', 'label_query_budget',
              'novelty_false_alarm_quantile', 'local_distance_quantile', 'agreement_similarity_quantile'):
        assert getattr(cfg, k) == lock['config'][k], f'{k} differs from the Phase 3D lock'
    return lock


DATASETS = {
    'realworld': dict(
        cache=PROJECT / 'data' / 'realworld' / 'cache' / 'realworld_forearm_acc_w128_s64_50hz.npz',
        manifest=PROJECT / 'artifacts' / 'phase3a' / 'phase3a_preregistered_manifest.json',
        models=PROJECT / 'artifacts' / 'phase3b' / 'folds',
        domains=lambda fold: 6,
        client_key='selection_subjects', eval_key='selection_subjects', limited_known_probe=True,
        status='Selection-only; test subjects never indexed.'),
    'pamap2': dict(
        cache=PROJECT / 'data' / 'pamap2' / 'cache' / 'pamap2_hand_acc_common8_w128_s64_50hz.npz',
        manifest=PROJECT / 'artifacts' / 'phase4a_fixed3' / 'phase4a_preregistered_manifest.json',
        models=PROJECT / 'artifacts' / 'phase4b' / 'folds',
        domains=lambda fold: len(fold['train_subjects']),
        client_key='integration_subjects', eval_key='test_subjects', limited_known_probe=False,
        status='Exploratory: test labels were opened in Phase 4B/4C.'),
    'ucihar': dict(
        cache=PROJECT / 'data' / 'ucihar' / 'cache' / 'ucihar_waist_totalacc_w128_s64_50hz.npz',
        manifest=PROJECT / 'artifacts' / 'phase5c' / 'phase5c_preregistered_manifest.json',
        models=PROJECT / 'artifacts' / 'phase5c' / 'folds',
        domains=lambda fold: len(fold['train_subjects']),
        client_key='integration_subjects', eval_key='test_subjects', limited_known_probe=False,
        status='Confirmatory: preregistered before download; test labels opened once in Phase 5C-C.'),
    'motionsense': dict(
        cache=PROJECT / 'data' / 'motionsense' / 'cache' / 'motionsense_pocket_useracc_w128_s64_50hz.npz',
        manifest=PROJECT / 'artifacts' / 'phase6' / 'phase6_preregistered_manifest.json',
        models=PROJECT / 'artifacts' / 'phase6' / 'folds',
        domains=lambda fold: len(fold['train_subjects']),
        client_key='integration_subjects', eval_key='test_subjects', limited_known_probe=False,
        status='Second confirmatory dataset: preregistered before download; tests generalization of the Phase 5C result.'),
}


class Block(nn.Module):
    def __init__(self, channels, width=32):
        super().__init__()
        self.a = nn.Conv1d(channels, width, 1, bias=False)
        self.b = nn.ModuleList([nn.Conv1d(width, width, k, padding=k // 2, bias=False) for k in (9, 19, 39)])
        self.p = nn.Sequential(nn.MaxPool1d(3, 1, 1), nn.Conv1d(channels, width, 1, bias=False))
        self.n = nn.BatchNorm1d(4 * width)
    def forward(self, q):
        z = self.a(q)
        return F.relu(self.n(torch.cat([f(z) for f in self.b] + [self.p(q)], 1)))


class Encoder(nn.Module):
    def __init__(self, domains):
        super().__init__()
        self.b1 = Block(4); self.b2 = Block(128)
        self.s = nn.Conv1d(4, 128, 1, bias=False); self.n = nn.BatchNorm1d(128)
        self.p = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64))
        self.decoder = nn.Conv1d(128, 4, 1)
        self.domain = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, domains))
    def features(self, q):
        return F.relu(self.n(self.b2(self.b1(q)) + self.s(q)))
    def forward(self, q):
        return F.normalize(self.p(self.features(q).mean(-1)), dim=1)


def adapt_checkpoint_keys(state):
    table = [('block1.bottleneck.', 'b1.a.'), ('block1.branches.', 'b1.b.'), ('block1.pool.', 'b1.p.'), ('block1.bn.', 'b1.n.'),
             ('block2.bottleneck.', 'b2.a.'), ('block2.branches.', 'b2.b.'), ('block2.pool.', 'b2.p.'), ('block2.bn.', 'b2.n.'),
             ('skip.', 's.'), ('project.', 'p.'), ('bn.', 'n.')]
    out = {}
    for key, value in state.items():
        for old, new in table: key = key.replace(old, new)
        out[key] = value
    return out


def load(path):
    try: return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError: return torch.load(path, map_location='cpu')


def unit(z): return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)


def encode(model, data, batch=512):
    out = []; model.eval()
    with torch.no_grad():
        for i in range(0, len(data), batch):
            out.append(model(torch.from_numpy(data[i:i + batch]).to(DEVICE)).cpu().numpy())
    return unit(np.concatenate(out))


class Frozen:
    """One dataset's windows plus per-fold frozen embeddings."""
    def __init__(self, name):
        self.name = name; ds = DATASETS[name]; self.ds = ds
        d = np.load(ds['cache'])
        self.x = d['x'].astype(np.float32); self.subjects = d['subject'].astype(int); self.activities = d['activity'].astype(str)
        man = json.loads(ds['manifest'].read_text(encoding='utf-8-sig'))
        self.known = tuple(man['known_activities']); self.novel = tuple(man['novel_activities'])
        self.vocabulary = self.known + self.novel; self.known_to_index = {a: i for i, a in enumerate(self.known)}
        self.folds = man['outer_folds']
    def normalized_input(self, stats, scale=1.0):
        x = self.x * float(scale)
        return np.concatenate([(x[:, :3] - stats['xyz_mean']) / stats['xyz_std'],
                               (x[:, 3:4] - stats['mag_mean']) / stats['mag_std']], 1).astype(np.float32)
    def embedding(self, fold):
        bundle = load(self.ds['models'] / f"fold_{fold['fold']}_models.pt")
        m = Encoder(self.ds['domains'](fold)).to(DEVICE)
        m.load_state_dict(adapt_checkpoint_keys(bundle['anchor_state_dict']), strict=True)
        # `input_scale` is absent in Phase 3B/4B bundles (m/s^2 caches); Phase 5C stores 9.80665 (g -> m/s^2).
        return encode(m, self.normalized_input(bundle['stats'], bundle.get('input_scale', 1.0)))
    def limited(self, subject_set, activity_set, n, seed):
        rng = np.random.default_rng(seed); out = []
        for s in subject_set:
            for a in activity_set:
                q = np.where((self.subjects == s) & (self.activities == a))[0]
                if len(q) > n: q = rng.choice(q, n, replace=False)
                out += q.tolist()
        return np.asarray(out, dtype=int)


def calibration(D, emb, fold, cfg=None):
    cfg = cfg or CFG
    fid = int(fold['fold']); train = tuple(fold['train_subjects']); sel = tuple(fold['selection_subjects'])
    tr = D.limited(train, D.known, cfg.probe_per_subject_activity, cfg.seed + 10 * fid)
    se = D.limited(sel, D.known, cfg.probe_per_subject_activity, cfg.seed + 10 * fid + 1)
    y = np.asarray([D.known_to_index[a] for a in D.activities[tr]])
    p = np.asarray([unit(emb[tr][y == k].mean(0, keepdims=True))[0] for k in range(len(D.known))])
    truth = np.asarray([D.known_to_index[a] for a in D.activities[se]]); score = emb[se] @ p.T
    novelty = float(np.quantile(score[np.arange(len(score)), truth], cfg.novelty_false_alarm_quantile))
    within = []; cross = []; rng = np.random.default_rng(cfg.seed + fid)
    for a in D.known:
        means = []
        for s in sel:
            q = np.where((D.subjects == s) & (D.activities == a))[0]
            if len(q) == 0: continue
            q = rng.choice(q, min(80, len(q)), replace=False); z = emb[q]
            within += (1 - z @ z.T)[np.triu_indices(len(z), 1)].tolist()
        for s in train:
            q = np.where((D.subjects == s) & (D.activities == a))[0]
            if len(q): means.append(unit(emb[q].mean(0, keepdims=True))[0])
        cross += [float(means[i] @ means[j]) for i in range(len(means)) for j in range(i + 1, len(means))]
    return p, {'novelty_threshold': novelty,
               'local_distance': float(np.quantile(within, cfg.local_distance_quantile)),
               'agreement_similarity': float(np.quantile(cross, cfg.agreement_similarity_quantile))}


def cluster(z, d):
    if len(z) < 2: return np.zeros(len(z), dtype=int)
    try: return AgglomerativeClustering(n_clusters=None, distance_threshold=d, metric='cosine', linkage='average').fit_predict(z)
    except TypeError: return AgglomerativeClustering(n_clusters=None, distance_threshold=d, affinity='cosine', linkage='average').fit_predict(z)


def ranked_windows(D, emb, p, client, cfg=None):
    cfg = cfg or CFG
    idx = np.where(D.subjects == client)[0]; nov = 1 - (emb[idx] @ p.T).max(1)
    order = np.argsort(nov)[-min(cfg.max_candidates_per_client, len(idx)):]
    return idx[order], nov[order]


def rows_from_clusters(D, emb, windows, labels, client, cfg=None):
    cfg = cfg or CFG; rows = []
    for lab in np.unique(labels):
        m = np.where(labels == lab)[0]
        if len(m) < cfg.min_local_cluster_size: continue
        w = windows[m]; sig = unit(emb[w].mean(0, keepdims=True))[0]; med = w[np.argmax(emb[w] @ sig)]
        v, n = np.unique(D.activities[w], return_counts=True)
        rows.append({'client': int(client), 'signature': sig, 'support': len(w), 'medoid': int(med),
                     'query_label': str(D.activities[med]), 'dominant': str(v[np.argmax(n)]), 'purity': float(n.max() / n.sum())})
    return rows


def local(D, emb, clients, p, c, distance=None, cfg=None):
    cfg = cfg or CFG; rows = []
    for s in clients:
        q, _ = ranked_windows(D, emb, p, s, cfg)
        rows += rows_from_clusters(D, emb, q, cluster(emb[q], c['local_distance'] if distance is None else distance), s, cfg)
    return rows


def mnn_groups(rows, t):
    if not rows: return []
    z = np.asarray([r['signature'] for r in rows]); sim = z @ z.T; best = {}
    for i, r in enumerate(rows):
        eligible = [j for j, s in enumerate(rows) if s['client'] != r['client'] and sim[i, j] >= t]
        if eligible: best[i] = max(eligible, key=lambda j: sim[i, j])
    groups = []; seen = set()
    for i, j in best.items():
        if best.get(j) == i and tuple(sorted((i, j))) not in seen:
            groups.append(sorted((i, j))); seen.add(tuple(sorted((i, j))))
    return groups


def group_candidates(D, rows, groups, require_agreement=True, cfg=None):
    cfg = cfg or CFG; out = []
    for g in groups:
        members = [rows[i] for i in g]; clients = set(r['client'] for r in members)
        if require_agreement and len(clients) < cfg.min_agreeing_clients: continue
        sig = unit(np.stack([r['signature'] for r in members]).mean(0, keepdims=True))[0]
        med = max(members, key=lambda r: sig @ r['signature'])['medoid']
        out.append({'signature': sig, 'label': str(D.activities[med]), 'clients': clients,
                    'support': sum(r['support'] for r in members),
                    'purity': float(np.average([r['purity'] for r in members], weights=[r['support'] for r in members])),
                    'dominant': [r['dominant'] for r in members]})
    return out


def spend_budget(D, candidates, order_key=None, cfg=None):
    """Query the top-B candidates; keep novel, de-duplicated answers. Every query is counted."""
    cfg = cfg or CFG
    queried = candidates if order_key is None else sorted(candidates, key=order_key, reverse=True)
    queried = queried[:cfg.label_query_budget]
    accepted = [r for r in queried if r['label'] in D.novel]; dedup = []; seen = set()
    for r in accepted:
        if r['label'] not in seen: dedup.append(r); seen.add(r['label'])
    return dedup, len(queried), [r['label'] for r in queried]


def policy_ranked_mnn(D, emb, clients, p, c, rows, seed, cfg=None, **_):
    cfg = cfg or CFG
    return spend_budget(D, group_candidates(D, rows, mnn_groups(rows, c['agreement_similarity']), cfg=cfg), lambda r: r['support'], cfg)


def policy_support_only(D, emb, clients, p, c, rows, seed, cfg=None, **_):
    cfg = cfg or CFG
    cands = group_candidates(D, rows, [[i] for i in range(len(rows))], require_agreement=False, cfg=cfg)
    return spend_budget(D, cands, lambda r: r['support'], cfg)


def policy_random_query(D, emb, clients, p, c, rows, seed, cfg=None, **_):
    cfg = cfg or CFG; rng = np.random.default_rng(seed)
    cands = group_candidates(D, rows, [[i] for i in range(len(rows))], require_agreement=False, cfg=cfg)
    pick = rng.permutation(len(cands))[:cfg.label_query_budget] if cands else []
    return spend_budget(D, [cands[i] for i in pick], cfg=cfg)


def policy_novelty_top_b(D, emb, clients, p, c, rows, seed, cfg=None, **_):
    cfg = cfg or CFG; pool = []
    for s in clients:
        q, nov = ranked_windows(D, emb, p, s, cfg)
        pool += [(float(n), int(w), int(s)) for w, n in zip(q, nov)]
    pool.sort(reverse=True)
    cands = [{'signature': emb[w], 'label': str(D.activities[w]), 'clients': {s}, 'support': 1, 'purity': float('nan'), 'dominant': [str(D.activities[w])]}
             for _, w, s in pool[:cfg.label_query_budget]]
    return spend_budget(D, cands, cfg=cfg)


def policy_centralized_oracle(D, emb, clients, p, c, rows, seed, cfg=None, **_):
    cfg = cfg or CFG
    pooled = np.concatenate([ranked_windows(D, emb, p, s, cfg)[0] for s in clients])
    central = rows_from_clusters(D, emb, pooled, cluster(emb[pooled], c['local_distance']), client=-1, cfg=cfg)
    cands = group_candidates(D, central, [[i] for i in range(len(central))], require_agreement=False, cfg=cfg)
    return spend_budget(D, cands, lambda r: r['support'], cfg)


def policy_component_mixture(D, emb, clients, p, c, rows, seed, cfg=None, **_):
    cfg = cfg or CFG; comp_rows = []
    for s in clients:
        q, _ = ranked_windows(D, emb, p, s, cfg)
        k_local = max(1, sum(1 for r in rows if r['client'] == s))
        k = int(min(cfg.gmm_max_components, max(2, k_local), max(2, len(q) // cfg.min_local_cluster_size)))
        gmm = GaussianMixture(n_components=k, covariance_type='diag', random_state=seed, n_init=1, reg_covar=1e-4).fit(emb[q])
        comp_rows += rows_from_clusters(D, emb, q, gmm.predict(emb[q]), s, cfg)
    groups = []; by_client = {}
    for i, r in enumerate(comp_rows): by_client.setdefault(r['client'], []).append(i)
    cl = sorted(by_client)
    for a in range(len(cl)):
        for b in range(a + 1, len(cl)):
            ia, ib = by_client[cl[a]], by_client[cl[b]]
            sim = np.asarray([[comp_rows[i]['signature'] @ comp_rows[j]['signature'] for j in ib] for i in ia])
            ri, ci = linear_sum_assignment(-sim)
            groups += [[ia[i], ib[j]] for i, j in zip(ri, ci) if sim[i, j] >= c['agreement_similarity']]
    return spend_budget(D, group_candidates(D, comp_rows, groups, cfg=cfg), lambda r: r['support'], cfg)


POLICIES = {
    'ranked_mnn': policy_ranked_mnn, 'random_query': policy_random_query, 'support_only': policy_support_only,
    'novelty_top_b': policy_novelty_top_b, 'component_mixture': policy_component_mixture, 'centralized_oracle': policy_centralized_oracle,
}


def known_prototypes(D, emb, fold, cfg=None):
    cfg = cfg or CFG; fid = int(fold['fold']); train = tuple(fold['train_subjects'])
    if D.ds['limited_known_probe']:
        known_idx = D.limited(train, D.known, cfg.probe_per_subject_activity, cfg.seed + 500 + fid)
    else:
        known_idx = np.where(np.isin(D.subjects, train) & np.isin(D.activities, D.known))[0]
    y = np.asarray([D.known_to_index[a] for a in D.activities[known_idx]])
    return np.asarray([unit(emb[known_idx][y == k].mean(0, keepdims=True))[0] for k in range(len(D.known))])


def evaluate(D, emb, fold, entries, cfg=None, eval_subjects=None):
    kp = known_prototypes(D, emb, fold, cfg)
    eval_idx = np.where(np.isin(D.subjects, tuple(eval_subjects if eval_subjects is not None else fold[D.ds['eval_key']])))[0]
    labels = list(D.known) + [r['label'] for r in entries]; proto = np.asarray([*kp] + [r['signature'] for r in entries])
    pred = np.asarray(labels)[(emb[eval_idx] @ proto.T).argmax(1)]
    f = precision_recall_fscore_support(D.activities[eval_idx], pred, labels=list(D.vocabulary), zero_division=0)[2]
    nk = len(D.known)
    per_class = {a: float(v) for a, v in zip(D.vocabulary, f)}
    return {'known_macro_f1': float(f[:nk].mean()), 'novel_macro_f1': float(f[nk:].mean()), 'combined_macro_f1': float(f.mean()),
            'per_class_f1': per_class}


def summarize(D, entries, queries, queried_labels, metrics, extra=None):
    row = {'promoted_entries': len(entries), 'label_queries': queries,
           'novel_recovery': len(set(r['label'] for r in entries)) / len(D.novel),
           'entry_purity': float(np.nanmean([r['purity'] for r in entries])) if entries else 0.,
           'known_label_queries': sum(1 for l in queried_labels if l in D.known),
           'wasted_queries': queries - len(entries)}
    per_class = metrics.pop('per_class_f1', None)
    row.update(metrics)
    if per_class: row.update({f'f1_{a}': v for a, v in per_class.items()})
    row.update(extra or {}); return row


def boot_ci(v, n=None, seed=None):
    n = n or CFG.bootstrap_resamples; seed = CFG.seed if seed is None else seed
    v = np.asarray(v, dtype=float); v = v[~np.isnan(v)]
    if len(v) == 0: return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed); m = rng.choice(v, (n, len(v)), replace=True).mean(1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))
