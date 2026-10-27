#!/usr/bin/env python3
"""Rebuild or check the notebook's deterministic four-module source bundle."""
import argparse
import ast
import base64
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    'arc_gremlin/__init__.py',
    'arc_gremlin/extended_solver.py',
    'arc_gremlin/saient_solver.py',
    'arc_gremlin/spatial_programs.py',
)
NOTEBOOK = 'kaggle/saient_arc_agi2_kaggle.ipynb'


def build_bundle(root=ROOT):
    """ZIP metadata and member order are fixed; only source bytes can vary."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for name in MODULES:
            member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            member.create_system = 3
            member.external_attr = 0o100644 << 16
            member.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(member, (Path(root)/name).read_bytes(), compresslevel=9)
    return base64.b64encode(output.getvalue()).decode('ascii')


def bundled_literal(notebook):
    values = []
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        for statement in ast.parse(''.join(cell['source'])).body:
            if (isinstance(statement, ast.Assign) and
                    any(isinstance(target, ast.Name) and target.id == 'BUNDLE_BASE64'
                        for target in statement.targets)):
                values.append(ast.literal_eval(statement.value))
    if len(values) != 1 or not isinstance(values[0], str):
        raise ValueError('expected one literal BUNDLE_BASE64 assignment')
    return values[0]


def update_notebook(root=ROOT, *, check=False):
    """Return whether the bundle already matches; check mode never writes."""
    root = Path(root)
    path = root/NOTEBOOK
    original = path.read_text()
    old = bundled_literal(json.loads(original))
    new = build_bundle(root)
    matches = old == new
    if not matches and not check:
        # Replace only the literal bytes, preserving every other notebook byte.
        if not old or original.count(old) != 1:
            raise ValueError('bundle literal must be nonempty and occur exactly once')
        path.write_text(original.replace(old, new, 1))
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true',
                        help='exit nonzero if the notebook needs regeneration; never write')
    args = parser.parse_args()
    matched = update_notebook(check=args.check)
    if args.check and not matched:
        parser.exit(1, 'Notebook bundle is stale; run python3 tools/build_kaggle_bundle.py\n')
    print('Notebook bundle matches source.' if matched else 'Notebook bundle regenerated from source.')


if __name__ == '__main__':
    main()
