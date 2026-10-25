# %% [markdown]
# # FedVocabHAR Phase 5C-A
#
# ## UCI-HAR download, cache construction, and gate G1 only
#
# This is the first contact with UCI-HAR. The preregistration in `artifacts/phase5c/phase5c_preregistered_manifest.json`
# was committed to GitHub before this notebook was run. This notebook:
#
# 1. downloads the official archive and records its SHA-256;
# 2. builds a cache in the same layout as the RealWorld and PAMAP2 caches (`x` of shape `(N, 4, 128)`:
#    total-acceleration XYZ plus magnitude, `subject`, `activity`);
# 3. evaluates **gate G1 only** (cache integrity) and prints nothing about labels beyond the subject × activity
#    count table required by G1.
#
# No embedding, no model, no metric involving activity separability is computed here.

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
import json, hashlib, zipfile, io, urllib.request
from pathlib import Path
import numpy as np, pandas as pd

PROJECT = Path('/content/drive/MyDrive/FedVocabHAR')
P5C = PROJECT / 'artifacts' / 'phase5c'; DATA = PROJECT / 'data' / 'ucihar'; RAW = DATA / 'raw'; CACHE_DIR = DATA / 'cache'
for d in (RAW, CACHE_DIR): d.mkdir(parents=True, exist_ok=True)
manifest = json.loads((P5C / 'phase5c_preregistered_manifest.json').read_text())
assert manifest['phase'] == 'Phase 5C' and manifest['window']['samples'] == 128
print('Preregistered:', manifest['registered_utc']); print('Known:', manifest['known_activities']); print('Novel:', manifest['novel_activities'])

# %% [markdown]
# ## Download and checksum

# %%
URL = manifest['official_source']; archive = RAW / 'human+activity+recognition+using+smartphones.zip'
if not archive.exists():
    print('downloading', URL); urllib.request.urlretrieve(URL, archive)
sha = hashlib.sha256(archive.read_bytes()).hexdigest(); print('sha256', sha, f'{archive.stat().st_size/1e6:.1f} MB')

# The UCI archive nests 'UCI HAR Dataset.zip' inside the outer zip.
outer = zipfile.ZipFile(archive); names = outer.namelist()
inner_name = next((n for n in names if n.lower().endswith('uci har dataset.zip')), None)
z = zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) if inner_name else outer
root = next(n for n in z.namelist() if n.endswith('activity_labels.txt')).rsplit('/', 1)[0]
print('dataset root inside archive:', root)

def read_txt(path):
    return np.loadtxt(io.BytesIO(z.read(f'{root}/{path}')))

labels = {int(k): v for k, v in (line.split() for line in z.read(f'{root}/activity_labels.txt').decode().strip().splitlines())}
print(labels)

# %% [markdown]
# ## Build cache
#
# Train and test folders are pooled; the shipped split is ignored because Phase 5C re-splits by subject. Windows are
# used exactly as shipped (128 samples, 50 % overlap). Magnitude is computed from total acceleration.

# %%
xs, subj, act, src, widx = [], [], [], [], []
for split in ('train', 'test'):
    tx = read_txt(f'{split}/Inertial Signals/total_acc_x_{split}.txt'); ty = read_txt(f'{split}/Inertial Signals/total_acc_y_{split}.txt'); tz = read_txt(f'{split}/Inertial Signals/total_acc_z_{split}.txt')
    s = read_txt(f'{split}/subject_{split}.txt').astype(int); y = read_txt(f'{split}/y_{split}.txt').astype(int)
    assert tx.shape == ty.shape == tz.shape == (len(s), 128) and len(y) == len(s)
    mag = np.sqrt(tx ** 2 + ty ** 2 + tz ** 2)
    xs.append(np.stack([tx, ty, tz, mag], 1).astype(np.float32)); subj.append(s); act.append(np.asarray([labels[v] for v in y])); src += [split] * len(s); widx.append(np.arange(len(s)))
x = np.concatenate(xs); subject = np.concatenate(subj); activity = np.concatenate(act); source = np.asarray(src); window_index = np.concatenate(widx)
cache = CACHE_DIR / 'ucihar_waist_totalacc_w128_s64_50hz.npz'
np.savez_compressed(cache, x=x, subject=subject, activity=activity, source_split=source, window_index=window_index)
print('cache', cache.name, x.shape, x.dtype, f'{cache.stat().st_size/1e6:.1f} MB')

# %% [markdown]
# ## Gate G1: cache integrity
#
# All 30 subjects × 6 activities present, at least 40 windows per cell, 4 channels, 128 samples. This is the only
# label-dependent table printed in Phase 5C-A.

# %%
counts = pd.crosstab(subject, activity).reindex(index=range(1, 31), columns=list(labels.values()), fill_value=0)
display(counts)
checks = {
    'subjects_present': int((counts.sum(1) > 0).sum()) == 30,
    'activities_present': int((counts.sum(0) > 0).sum()) == 6,
    'all_cells_nonempty': bool((counts > 0).all().all()),
    'min_windows_per_cell_ge_40': int(counts.values.min()) >= 40,
    'channels_eq_4': x.shape[1] == 4,
    'window_len_eq_128': x.shape[2] == 128,
    'finite': bool(np.isfinite(x).all()),
    'fold_subjects_cover_all_30': sorted(sum((f['test_subjects'] for f in manifest['outer_folds']), [])) == list(range(1, 31)),
}
g1_pass = all(checks.values())
record = {'phase': 'Phase 5C-A', 'gate': 'G1', 'pass': g1_pass, 'checks': checks, 'archive_sha256': sha, 'archive_bytes': archive.stat().st_size,
          'n_windows': int(len(x)), 'min_cell_windows': int(counts.values.min()), 'max_cell_windows': int(counts.values.max()),
          'cache': str(cache.relative_to(PROJECT)), 'activity_labels': labels, 'note': 'No embedding or separability metric computed. Counts table is the only label-dependent output.'}
(P5C / 'phase5c_a_g1_record.json').write_text(json.dumps(record, indent=2)); counts.to_csv(P5C / 'phase5c_a_subject_activity_counts.csv')
print(json.dumps(record, indent=2))
if not g1_pass:
    print('\nG1 FAILED. Record the failure in artifacts/phase5c/phase5c_deviations.md before proceeding to Phase 5C-B.')

# %% [markdown]
# ## Next step
#
# If G1 passed, run `FedVocabHAR_Phase5C_B_Encoders.ipynb` (per-fold training on train subjects only).
# If G1 failed on the minimum-window criterion only, the preregistration requires an explicit entry in
# `phase5c_deviations.md` describing the shortfall before any further step.
