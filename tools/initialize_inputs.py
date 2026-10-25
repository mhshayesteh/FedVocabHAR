"""Prepare untracked runtime inputs without overwriting existing files."""
from pathlib import Path
import argparse
import shutil

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    'realworld_protocol.json': 'phase3a/phase3a_preregistered_manifest.json',
    'pamap2_protocol.json': 'phase4a_fixed3/phase4a_preregistered_manifest.json',
    'ucihar_protocol.json': 'phase5c/phase5c_preregistered_manifest.json',
    'motionsense_protocol.json': 'phase6/phase6_preregistered_manifest.json',
    'encoder_recipe.json': 'phase3b/phase3b_decision.json',
    'promotion_policy.json': 'phase3d/phase3d_policy_lock.json',
}
NOTES = {
    'phase5c/phase5c_deviations.md': '''# UCI-HAR protocol conventions

## D1. Minimum-window count shortfall
Four original subject/activity cells contain 36-39 windows rather than 40.
Use shipped windows and retain the failed data-integrity flag.

## D2. Input units
Convert native g units to m/s^2 with input_scale = 9.80665.

## D3. Checkpoint rule
Keep the final epoch of each training stage; no held-out checkpoint selection.
''',
    'phase6/phase6_deviations.md': '''# MotionSense protocol conventions

## D1. Minimum-window count shortfall
The original subject 13 upstairs cell contains 39 windows rather than 40.
Use shipped windowing and retain the failed data-integrity flag.
''',
}

def initialize(project):
    project = Path(project).resolve()
    created = []
    for source, relative in INPUTS.items():
        target = project / 'artifacts' / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / 'config' / source, target)
            created.append(target)
    for relative, text in NOTES.items():
        target = project / 'artifacts' / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding='utf-8')
            created.append(target)
    # Supply the import path used by unchanged original Colab cells.
    for source in (ROOT / 'notebooks' / 'src').glob('*.py'):
        target = project / 'colab' / 'src' / source.name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            created.append(target)
    return created

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=ROOT,
                        help='Runtime directory; defaults to this checkout')
    args = parser.parse_args()
    for path in initialize(args.project):
        print(path)
