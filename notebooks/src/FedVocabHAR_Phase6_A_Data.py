# %% [markdown]
# # FedVocabHAR Phase 6-A
#
# ## MotionSense download, cache construction, and gate G1 only
#
# This is the first contact with MotionSense. The preregistration in `artifacts/phase6/phase6_preregistered_manifest.json`
# was committed to GitHub (`f7ffaba`) before this notebook was run. This notebook:
#
# 1. downloads the official `A_DeviceMotion_data.zip` archive and records its SHA-256;
# 2. builds a cache in the same layout as every prior Phase 5 cache (`x` of shape `(N, 4, 128)`:
#    userAcceleration XYZ plus magnitude, in m/s^2 after the preregistered `input_scale = 9.80665`
#    g->m/s^2 conversion, applied here at cache-build time exactly as declared in the manifest's
#    `documented_unit_convention` field);
# 3. evaluates **gate G1 only** (cache integrity) and prints nothing about labels beyond the subject x
#    activity count table required by G1.
#
# No embedding, no model, no metric involving activity separability is computed here.

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
import json, hashlib, zipfile, io, re, urllib.request
from pathlib import Path
import numpy as np, pandas as pd

PROJECT = Path('/content/drive/MyDrive/FedVocabHAR')
P6 = PROJECT / 'artifacts' / 'phase6'; DATA = PROJECT / 'data' / 'motionsense'; RAW = DATA / 'raw'; CACHE_DIR = DATA / 'cache'
for d in (RAW, CACHE_DIR): d.mkdir(parents=True, exist_ok=True)
manifest = json.loads((P6 / 'phase6_preregistered_manifest.json').read_text(encoding='utf-8-sig'))
assert manifest['phase'] == 'Phase 6' and manifest['window']['samples'] == 128
KNOWN = tuple(manifest['known_activities']); NOVEL = tuple(manifest['novel_activities'])
print('Preregistered:', manifest['registered_utc']); print('Known:', KNOWN); print('Novel:', NOVEL)

# %% [markdown]
# ## Download and checksum

# %%
URL = 'https://raw.githubusercontent.com/mmalekzadeh/motion-sense/master/data/A_DeviceMotion_data.zip'
archive = RAW / 'A_DeviceMotion_data.zip'
if not archive.exists():
    print('downloading', URL); urllib.request.urlretrieve(URL, archive)
sha = hashlib.sha256(archive.read_bytes()).hexdigest(); print('sha256', sha, f'{archive.stat().st_size/1e6:.1f} MB')
z = zipfile.ZipFile(archive); names = z.namelist()

# %% [markdown]
# ## Build cache
#
# Trial folders are named `{activity}_{trial}` (e.g. `wlk_7`); each contains one CSV per subject
# (`sub_{code}.csv`, code 1-24) with a leading unnamed index column plus
# `attitude.{roll,pitch,yaw}`, `gravity.{x,y,z}`, `rotationRate.{x,y,z}`, `userAcceleration.{x,y,z}`.
# Only `userAcceleration.{x,y,z}` is used, per the manifest's `sensor` field. Each trial recording is
# segmented independently into 128-sample windows at 50 % overlap (stride 64); windows are never built
# across a trial boundary. Magnitude is computed after the g->m/s^2 scale is applied.

# %%
ACT_TRIALS = {'dws': [1, 2, 11], 'ups': [3, 4, 12], 'wlk': [7, 8, 15], 'jog': [9, 16], 'std': [6, 14], 'sit': [5, 13]}
assert set(ACT_TRIALS) == set(KNOWN) | set(NOVEL), 'activity codes must match the preregistered known+novel set'
SCALE = float(manifest['documented_unit_convention'].split('input_scale of ')[1].split(' ')[0]) if 'input_scale of' in manifest['documented_unit_convention'] else 9.80665
print('input_scale (g -> m/s^2), applied at cache-build time per manifest:', SCALE)

# The archive's top-level folder name is 'A_DeviceMotion_data/'; some mirrors nest it one level deeper.
root = next((m.group(1) for n in names if (m := re.search(r'(.*A_DeviceMotion_data)/[a-z]+_\d+/sub_\d+\.csv$', n))), None)
assert root is not None, 'could not locate A_DeviceMotion_data trial folders inside the archive'
print('archive root prefix for trial folders:', root)

xs, subj, act, trial_list, found_subjects = [], [], [], [], set()
WIN, STRIDE = manifest['window']['samples'], manifest['window']['stride']
for a, trials in ACT_TRIALS.items():
    for trial in trials:
        for code in range(1, 25):
            member = f'{root}/{a}_{trial}/sub_{code}.csv'
            if member not in names: continue
            df = pd.read_csv(io.BytesIO(z.read(member)))
            df = df.drop(columns=[c for c in df.columns if c.startswith('Unnamed')], errors='ignore')
            xyz = df[['userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z']].to_numpy(dtype=np.float32) * SCALE
            n = len(xyz)
            if n < WIN: continue
            starts = np.arange(0, n - WIN + 1, STRIDE)
            for st in starts:
                w = xyz[st:st + WIN].T  # (3, 128)
                mag = np.sqrt((w ** 2).sum(0, keepdims=True))
                xs.append(np.concatenate([w, mag], 0).astype(np.float32))
                subj.append(code); act.append(a); trial_list.append(trial)
            found_subjects.add(code)
assert len(xs) > 0, 'no windows were built; check the archive layout against ACT_TRIALS'
x = np.stack(xs); subject = np.asarray(subj, dtype=int); activity = np.asarray(act); trial_arr = np.asarray(trial_list, dtype=int)
cache = CACHE_DIR / 'motionsense_pocket_useracc_w128_s64_50hz.npz'
np.savez_compressed(cache, x=x, subject=subject, activity=activity, trial=trial_arr)
print('cache', cache.name, x.shape, x.dtype, f'{cache.stat().st_size/1e6:.1f} MB')
print('subjects found in archive:', sorted(found_subjects))

# %% [markdown]
# ## Gate G1: cache integrity
#
# All 24 subjects x 6 activities present, at least 40 windows per cell, 4 channels, 128 samples. Any shortfall
# against the manifest's `subject_id_assumption` (24 subjects numbered 1-24) must be logged as a deviation in
# `phase6_deviations.md` before Phase 6-B, exactly as D1 was logged for UCI-HAR.

# %%
labels_order = list(KNOWN) + list(NOVEL)
counts = pd.crosstab(subject, activity).reindex(index=range(1, 25), columns=labels_order, fill_value=0)
display(counts)
fold_subjects_union = sorted(set(sum((f['train_subjects'] + f['selection_subjects'] + f['integration_subjects'] + f['test_subjects'] for f in manifest['outer_folds']), [])))
checks = {
    'subjects_present_eq_24': int((counts.sum(1) > 0).sum()) == 24,
    'subject_ids_are_1_to_24': found_subjects == set(range(1, 25)),
    'activities_present_eq_6': int((counts.sum(0) > 0).sum()) == 6,
    'all_cells_nonempty': bool((counts > 0).all().all()),
    'min_windows_per_cell_ge_40': int(counts.values.min()) >= 40,
    'channels_eq_4': x.shape[1] == 4,
    'window_len_eq_128': x.shape[2] == 128,
    'finite': bool(np.isfinite(x).all()),
    'fold_subjects_cover_all_24': fold_subjects_union == list(range(1, 25)),
}
g1_pass = all(checks.values())
record = {'phase': 'Phase 6-A', 'gate': 'G1', 'pass': g1_pass, 'checks': checks, 'archive_sha256': sha, 'archive_bytes': archive.stat().st_size,
          'input_scale_applied_at_cache_build': SCALE, 'n_windows': int(len(x)), 'min_cell_windows': int(counts.values.min()), 'max_cell_windows': int(counts.values.max()),
          'cache': str(cache.relative_to(PROJECT)), 'activity_order': labels_order,
          'note': 'No embedding or separability metric computed. Counts table is the only label-dependent output.'}
(P6 / 'phase6_a_g1_record.json').write_text(json.dumps(record, indent=2)); counts.to_csv(P6 / 'phase6_a_subject_activity_counts.csv')
print(json.dumps(record, indent=2))
if not g1_pass:
    print('\nG1 FAILED on:', [k for k, v in checks.items() if not v], '- record this in artifacts/phase6/phase6_deviations.md before proceeding to Phase 6-B.')

# %% [markdown]
# ## Next step
#
# If G1 passed (or failed only on the minimum-window criterion, mirroring UCI-HAR's D1), run
# `FedVocabHAR_Phase6_B_Encoders.ipynb` (per-fold training on train subjects only). Any other G1 failure
# (subject count/ID mismatch, missing activity) must be written up as a deviation before continuing, and may
# require re-deriving the folds from the actual subject set -- in which case this is no longer a strict
# confirmatory test and must be reported as such.
