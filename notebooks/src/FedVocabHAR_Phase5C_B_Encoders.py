# %% [markdown]
# # FedVocabHAR Phase 5C-B: Per-fold encoder training on UCI-HAR (training subjects only)
#
# Confirmatory dataset. This notebook trains one frozen-recipe encoder per outer fold using **only the
# `train_subjects`** of `artifacts/phase5c/phase5c_preregistered_manifest.json`. It follows the Phase 3B recipe
# and the Phase 4B third-dataset procedure with three preregistration-driven differences, recorded in
# `artifacts/phase5c/phase5c_deviations.md` (D1-D3):
#
# * D1: G1 failed only on the >=40-windows-per-cell check (four downstairs cells at 36-39). Proceed as shipped.
# * D2: input is converted from g to m/s^2 (`input_scale = 9.80665`) so augmentation noise has the same
#   relative strength as in the frozen recipe.
# * D3: **no checkpoint selection**: the final epoch of each stage is kept. Only training-subject diagnostics
#   are printed. Selection, integration, and test subjects are never indexed here.
#
# Output: `artifacts/phase5c/folds/fold_{i}_models.pt` (+ `fold_{i}_train_complete.json`) in the key format
# read by `colab/src/phase5_common.py`.

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
import sys; sys.path.insert(0, '/content/drive/MyDrive/FedVocabHAR/colab/src')
import phase5_common as P5
from phase5_common import PROJECT, DEVICE, Encoder, seed_all, unit
from dataclasses import dataclass, asdict
from pathlib import Path
import copy, json, time, warnings
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.autograd import Function
from sklearn.metrics import precision_recall_fscore_support
warnings.filterwarnings('ignore')

@dataclass(frozen=True)
class Recipe:  # Phase 3B / 4B frozen recipe; verified against artifacts/phase3b/phase3b_decision.json below.
    seed: int = 42
    embedding_dim: int = 64
    ssl_per_subject_activity: int = 300
    batch_size: int = 192
    ssl_epochs: int = 20
    ssl_learning_rate: float = 8e-4
    ssl_weight_decay: float = 1e-4
    temperature: float = .12
    reconstruction_weight: float = .30
    domain_weight: float = .08
    variance_weight: float = .08
    mask_fraction: float = .25
    support_per_class: int = 8
    query_per_class: int = 12
    episodes_per_epoch: int = 36
    episodic_epochs: int = 20
    episodic_learning_rate: float = 2e-4
    episodic_weight_decay: float = 1e-5
    prototype_temperature: float = .10
    anchor_weight: float = .25
    input_scale: float = 9.80665  # D2: UCI-HAR total_acc is in g; frozen recipe was fixed on m/s^2 data.

R = Recipe(); P5C = PROJECT / 'artifacts' / 'phase5c'; FOLD_DIR = P5C / 'folds'; FOLD_DIR.mkdir(parents=True, exist_ok=True)
manifest = json.loads((P5C / 'phase5c_preregistered_manifest.json').read_text(encoding='utf-8-sig'))
g1 = json.loads((P5C / 'phase5c_a_g1_record.json').read_text(encoding='utf-8-sig'))
deviations = (P5C / 'phase5c_deviations.md').read_text(encoding='utf-8-sig')
assert manifest['phase'] == 'Phase 5C' and len(manifest['outer_folds']) == 5
assert '## D1.' in deviations and '## D2.' in deviations and '## D3.' in deviations, 'record deviations before training'
failed_g1 = [k for k, v in g1['checks'].items() if not v]
assert failed_g1 == ['min_windows_per_cell_ge_40'], f'unexpected G1 failures {failed_g1}; stop and record them'
print('G1 record: pass =', g1['pass'], '| failed checks:', failed_g1, '(covered by deviation D1)')

# Recipe guard against the Phase 3B decision record (numbers must match the frozen recipe).
p3b = json.loads((PROJECT / 'artifacts' / 'phase3b' / 'phase3b_decision.json').read_text(encoding='utf-8-sig'))
cfg3b = p3b.get('config', p3b)
for k in ('ssl_epochs', 'ssl_learning_rate', 'temperature', 'reconstruction_weight', 'domain_weight', 'variance_weight',
          'mask_fraction', 'support_per_class', 'query_per_class', 'episodes_per_epoch', 'episodic_epochs',
          'episodic_learning_rate', 'prototype_temperature', 'anchor_weight', 'batch_size', 'embedding_dim'):
    if k in cfg3b: assert getattr(R, k) == cfg3b[k], f'{k}: recipe {getattr(R, k)} != Phase 3B {cfg3b[k]}'
print('Recipe matches Phase 3B. device:', DEVICE); print(json.dumps(asdict(R), indent=1))

# %% [markdown]
# ## Data (training subjects only are ever indexed below)

# %%
d = np.load(P5C.parent.parent / 'data' / 'ucihar' / 'cache' / 'ucihar_waist_totalacc_w128_s64_50hz.npz')
x_all = d['x'].astype(np.float32) * R.input_scale; subjects = d['subject'].astype(int); activities = d['activity'].astype(str)
KNOWN = tuple(manifest['known_activities']); k2i = {a: i for i, a in enumerate(KNOWN)}
assert x_all.shape[1:] == (4, 128) and set(np.unique(subjects)) == set(range(1, 31))
print('cache', x_all.shape, '| known', KNOWN, '| novel (never indexed here):', manifest['novel_activities'])

def limited(subject_set, activity_set, per_group, seed):
    rng = np.random.default_rng(seed); out = []
    for s in subject_set:
        for a in activity_set:
            idx = np.where((subjects == s) & (activities == a))[0]
            if len(idx) > per_group: idx = rng.choice(idx, per_group, replace=False)
            out.extend(idx.tolist())
    return np.asarray(out, dtype=int)

# %% [markdown]
# ## Frozen recipe components (verbatim from Phase 4B, adapted to `phase5_common.Encoder`)

# %%
class GradientReverse(Function):
    @staticmethod
    def forward(ctx, q, lambd): ctx.lambd = lambd; return q.view_as(q)
    @staticmethod
    def backward(ctx, grad): return -ctx.lambd * grad, None

def ntxent(z1, z2, temp):
    z = torch.cat([z1, z2], 0); sim = z @ z.T / temp; n = len(z1); eye = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9); target = torch.cat([torch.arange(n, 2 * n, device=z.device), torch.arange(n, device=z.device)])
    return F.cross_entropy(sim, target)

def variance_loss(z): return F.relu(1 - torch.sqrt(z.var(0) + 1e-4)).mean()

def random_rotation(batch):
    q = F.normalize(torch.randn(batch, 4, device=DEVICE), dim=1); w, a, b, c = q.unbind(1)
    return torch.stack([1 - 2 * (b * b + c * c), 2 * (a * b - c * w), 2 * (a * c + b * w), 2 * (a * b + c * w), 1 - 2 * (a * a + c * c),
                        2 * (b * c - a * w), 2 * (a * c - b * w), 2 * (b * c + a * w), 1 - 2 * (a * a + b * b)], 1).view(batch, 3, 3)

def make_inputs(raw, stats):
    xyz = (raw[:, :3] - stats['xyz_mean']) / stats['xyz_std']; mag = (raw[:, 3:4] - stats['mag_mean']) / stats['mag_std']
    return np.concatenate([xyz, mag], 1).astype(np.float32)

def sensor_view(raw, stats, strong=True):
    q = raw[:, :3].clone(); q = torch.bmm(random_rotation(len(q)), q)
    q = q * (1 + torch.randn(len(q), 1, 1, device=q.device) * (.12 if strong else .06)); q = q + torch.randn_like(q) * (.035 if strong else .02)
    shifts = torch.randint(-8 if strong else -4, 9 if strong else 5, (len(q),), device=q.device)
    time_idx = (torch.arange(q.size(-1), device=q.device)[None, :] - shifts[:, None]) % q.size(-1)
    q = torch.gather(q, 2, time_idx[:, None, :].expand(-1, 3, -1))
    if strong:
        length = q.size(-1); keep = int(torch.randint(int(.82 * length), length + 1, (1,), device=q.device).item())
        start = int(torch.randint(0, length - keep + 1, (1,), device=q.device).item())
        q = F.interpolate(q[:, :, start:start + keep], size=length, mode='linear', align_corners=False)
    t = lambda k: torch.as_tensor(stats[k], device=q.device)
    xyz = (q - t('xyz_mean')) / t('xyz_std'); mag = torch.sqrt((q * q).sum(1, keepdim=True) + 1e-8); mag = (mag - t('mag_mean')) / t('mag_std')
    return torch.cat([xyz, mag], 1)

def encode(model, data, batch=512):
    out = []; model.eval()
    with torch.no_grad():
        for i in range(0, len(data), batch): out.append(model(torch.from_numpy(data[i:i + batch]).to(DEVICE)).cpu().numpy())
    return unit(np.concatenate(out))

def train_subject_diagnostic(model, model_input, train_subjects, seed):
    """Prototype macro-F1 on training subjects only (fit and evaluate on the same subjects; a sanity check, not a selection signal)."""
    idx = limited(train_subjects, KNOWN, 220, seed); z = encode(model, model_input[idx]); y = np.asarray([k2i[a] for a in activities[idx]])
    proto = np.asarray([unit(z[y == k].mean(0, keepdims=True))[0] for k in range(len(KNOWN))]); pred = (z @ proto.T).argmax(1)
    return float(precision_recall_fscore_support(y, pred, labels=list(range(len(KNOWN))), zero_division=0)[2].mean())

# %% [markdown]
# ## Stage 1: self-supervised training (final epoch kept; D3)

# %%
def train_ssl(fold_id, train_subjects, stats, model_input):
    seed = R.seed + 100 * fold_id; seed_all(seed); ssl_idx = limited(train_subjects, KNOWN, R.ssl_per_subject_activity, seed)
    s2d = {s: i for i, s in enumerate(train_subjects)}; model = Encoder(len(train_subjects)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=R.ssl_learning_rate, weight_decay=R.ssl_weight_decay); rng = np.random.default_rng(seed); history = []
    for epoch in range(1, R.ssl_epochs + 1):
        model.train(); losses = []; order = rng.permutation(ssl_idx)
        for start in range(0, len(order), R.batch_size):
            idx = order[start:start + R.batch_size]
            if len(idx) < 16: continue
            raw = torch.from_numpy(x_all[idx]).to(DEVICE); v1 = sensor_view(raw, stats, True); v2 = sensor_view(raw, stats, True)
            opt.zero_grad(); z1 = model(v1); z2 = model(v2)
            loss = ntxent(z1, z2, R.temperature) + R.variance_weight * (variance_loss(z1) + variance_loss(z2))
            domains = torch.tensor([s2d[int(s)] for s in subjects[idx]], device=DEVICE)
            loss = loss + R.domain_weight * F.cross_entropy(model.domain(GradientReverse.apply((z1 + z2) / 2, 1.0)), domains)
            clean = sensor_view(raw, stats, False); length = max(1, int(clean.size(-1) * R.mask_fraction))
            starts = torch.randint(0, clean.size(-1) - length + 1, (len(clean),), device=DEVICE); timeline = torch.arange(clean.size(-1), device=DEVICE)[None, :]
            time_mask = (timeline >= starts[:, None]) & (timeline < (starts + length)[:, None]); masked = clean.masked_fill(time_mask[:, None, :], 0)
            recon = model.decoder(model.features(masked)); mask = time_mask[:, None, :].expand_as(clean).to(clean.dtype)
            rec = ((recon - clean).pow(2) * mask).sum() / mask.sum().clamp_min(1)
            loss = loss + R.reconstruction_weight * rec; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); losses.append(float(loss.detach().cpu()))
        row = {'fold': fold_id, 'stage': 'ssl', 'epoch': epoch, 'loss': float(np.mean(losses))}
        if epoch % 4 == 0 or epoch == R.ssl_epochs:
            row['train_subject_prototype_f1'] = train_subject_diagnostic(model, model_input, train_subjects, R.seed + 1000 * fold_id)
            print(f'fold {fold_id} SSL epoch {epoch:2d} loss {row["loss"]:.4f} train-subject F1 {row["train_subject_prototype_f1"]:.4f}')
        history.append(row)
    model.eval(); return model, history

# %% [markdown]
# ## Stage 2: episodic-anchor fine-tuning (final epoch kept; D3)

# %%
def train_episodic(fold_id, ssl_model, train_subjects, model_input):
    seed = R.seed + 100 * fold_id + 17; seed_all(seed); anchor = copy.deepcopy(ssl_model).to(DEVICE).eval()
    for p in anchor.parameters(): p.requires_grad = False
    model = copy.deepcopy(ssl_model).to(DEVICE); opt = torch.optim.AdamW(model.parameters(), lr=R.episodic_learning_rate, weight_decay=R.episodic_weight_decay)
    rng = np.random.default_rng(seed); lookup = {(a, s): np.where((activities == a) & (subjects == s))[0] for a in KNOWN for s in train_subjects}; history = []
    for epoch in range(1, R.episodic_epochs + 1):
        model.train(); losses = []
        for _ in range(R.episodes_per_epoch):
            si, qi, sy, qy = [], [], [], []
            for label, a in enumerate(KNOWN):
                ss, qs = rng.choice(train_subjects, 2, replace=False); sidx = lookup[(a, int(ss))]; qidx = lookup[(a, int(qs))]
                si.extend(rng.choice(sidx, R.support_per_class, replace=len(sidx) < R.support_per_class)); qi.extend(rng.choice(qidx, R.query_per_class, replace=len(qidx) < R.query_per_class))
                sy.extend([label] * R.support_per_class); qy.extend([label] * R.query_per_class)
            sx = torch.from_numpy(model_input[np.asarray(si)]).to(DEVICE); qx = torch.from_numpy(model_input[np.asarray(qi)]).to(DEVICE)
            sy = torch.tensor(sy, device=DEVICE); qy = torch.tensor(qy, device=DEVICE)
            opt.zero_grad(); zs = model(sx); zq = model(qx)
            prototypes = torch.stack([F.normalize(zs[sy == label].mean(0), dim=0) for label in range(len(KNOWN))])
            loss = F.cross_entropy(zq @ prototypes.T / R.prototype_temperature, qy)
            with torch.no_grad(): anchor_z = torch.cat([anchor(sx), anchor(qx)], 0)
            current = torch.cat([zs, zq], 0); loss = loss + R.anchor_weight * (1 - (current * anchor_z).sum(1)).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); losses.append(float(loss.detach().cpu()))
        row = {'fold': fold_id, 'stage': 'episodic_anchor', 'epoch': epoch, 'loss': float(np.mean(losses))}
        if epoch % 4 == 0 or epoch == R.episodic_epochs:
            row['train_subject_prototype_f1'] = train_subject_diagnostic(model, model_input, train_subjects, R.seed + 1000 * fold_id)
            print(f'fold {fold_id} anchor epoch {epoch:2d} loss {row["loss"]:.4f} train-subject F1 {row["train_subject_prototype_f1"]:.4f}')
        history.append(row)
    model.eval(); return model, history

# %% [markdown]
# ## Run all five folds (resumable)

# %%
for fold in manifest['outer_folds']:
    fid = int(fold['fold']); ckpt = FOLD_DIR / f'fold_{fid}_models.pt'; done = FOLD_DIR / f'fold_{fid}_train_complete.json'
    if ckpt.exists() and done.exists(): print('fold', fid, 'already complete; skipping'); continue
    train_subjects = tuple(int(s) for s in fold['train_subjects'])
    roles = [set(fold[k]) for k in ('train_subjects', 'selection_subjects', 'integration_subjects', 'test_subjects')]
    assert all(roles[i].isdisjoint(roles[j]) for i in range(4) for j in range(i + 1, 4)) and set.union(*roles) == set(range(1, 31))
    t0 = time.time()
    stat_idx = np.where(np.isin(subjects, train_subjects) & np.isin(activities, KNOWN))[0]; raw = x_all[stat_idx]
    stats = {'xyz_mean': raw[:, :3].mean((0, 2), keepdims=True).astype(np.float32), 'xyz_std': (raw[:, :3].std((0, 2), keepdims=True) + 1e-6).astype(np.float32),
             'mag_mean': raw[:, 3:4].mean((0, 2), keepdims=True).astype(np.float32), 'mag_std': (raw[:, 3:4].std((0, 2), keepdims=True) + 1e-6).astype(np.float32)}
    model_input = make_inputs(x_all, stats)  # normalized for all windows; only train_subjects rows are indexed below
    ssl_model, ssl_hist = train_ssl(fid, train_subjects, stats, model_input)
    anchor_model, anchor_hist = train_episodic(fid, ssl_model, train_subjects, model_input)
    torch.save({'fold': fold, 'recipe': asdict(R), 'stats': stats, 'input_scale': R.input_scale,
                'checkpoint_rule': 'final epoch of each stage; no selection (deviation D3)',
                'ssl_state_dict': {k: v.cpu() for k, v in ssl_model.state_dict().items()},
                'anchor_state_dict': {k: v.cpu() for k, v in anchor_model.state_dict().items()}}, ckpt)
    done.write_text(json.dumps({'fold': fid, 'roles': fold, 'train_subjects_only': True, 'history': ssl_hist + anchor_hist,
                                'seconds': time.time() - t0, 'device': str(DEVICE)}, indent=2))
    print(f'completed fold {fid} in {time.time() - t0:.0f}s'); torch.cuda.empty_cache()

# %% [markdown]
# ## Load-back check and record

# %%
D = P5.Frozen('ucihar'); records = []
for fold in D.folds:
    emb = D.embedding(fold)  # exercises the same loader Phase 5C-C will use (input_scale applied)
    assert emb.shape == (len(D.x), 64) and np.isfinite(emb).all()
    tr = np.isin(D.subjects, fold['train_subjects']) & np.isin(D.activities, D.known)
    y = np.asarray([D.known_to_index[a] for a in D.activities[tr]]); z = emb[tr]
    proto = np.asarray([unit(z[y == k].mean(0, keepdims=True))[0] for k in range(len(D.known))])
    f1 = float(precision_recall_fscore_support(y, (z @ proto.T).argmax(1), labels=list(range(len(D.known))), zero_division=0)[2].mean())
    records.append({'fold': int(fold['fold']), 'train_subject_prototype_known_f1': f1}); print('fold', fold['fold'], 'train-subject known F1', round(f1, 4))
hist = pd.DataFrame([r for f in D.folds for r in json.loads((FOLD_DIR / f"fold_{f['fold']}_train_complete.json").read_text())['history']])
hist.to_csv(P5C / 'phase5c_b_training_history.csv', index=False)
(P5C / 'phase5c_b_record.json').write_text(json.dumps({
    'phase': 'Phase 5C-B', 'trained_on': 'train_subjects only', 'checkpoint_rule': 'final epoch (deviation D3)', 'input_scale': R.input_scale,
    'g1_failed_checks': failed_g1, 'deviations_cited': ['D1', 'D2', 'D3'], 'recipe': asdict(R), 'folds': records,
    'note': 'No selection, integration, or test subject was indexed. No metric on non-training subjects was computed.'}, indent=2))
print('Phase 5C-B complete. Next: FedVocabHAR_Phase5C_C_Confirmatory.ipynb (opens test labels exactly once).')
