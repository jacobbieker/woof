"""Regenerate the GFDL Eta column and CO2 initialization units at one storage tier.

    GFDLETA_MAXL=256 python tools/gfdleta_wrf461_oracle/build_tier.py WRF F2C WORK OUTPUT_ROOT

runs the pinned lowering (lower_column.py, round_parameters.py, f2c,
assemble_column.py, make_column_wrapper.py; lower_init.py, f2c,
assemble_init.py) with every per-column array bounded by ``GFDLETA_MAXL``.
WRF's ``NL=81`` ozone climatology stays 81 at every tier: it is the fixed
GFDL pressure grid OZON2D interpolates from, not a model-level bound.

At the shipped tier (``GFDLETA_MAXL`` unset or 81) the outputs are the
committed ``gfdleta_column.cuh``/``.cu``/layout and ``gfdleta_co2_init.cuh``;
writing them into a scratch ``OUTPUT_ROOT`` and comparing proves the
generator still reproduces the shipped text (tests/test_gfdleta_256_levels.py
records the expected digests). At 256 it writes the ``_l256`` tier files.
The hand-written CO2 launch unit ``gfdleta_co2_init.cu`` is carried to the
tier by explicit, counted storage-size substitutions only.
"""
from pathlib import Path
import importlib.util, json, os, re, shutil, subprocess, sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MAXL = int(os.environ.get('GFDLETA_MAXL', '81'))
TIER = '' if MAXL == 81 else f'_l{MAXL}'
NOINLINE = ('co2in1_', 'co2ins_', 'co2int_', 'quadsr_', 'co2o3_', 'co2ptz_', 'sigp_')
NOINLINE_NOTE = ('// Keep large initialization subtrees out of callers to bound NVRTC memory.\n'
                 '// Only call boundaries change; the transcribed arithmetic stays in place.\n')


def run(*args, cwd=None):
    subprocess.run([str(a) for a in args], cwd=cwd, check=True)


def lowered(*args):
    """Run one lowering. Its chatter is not kept; a failure stops the build."""
    r = subprocess.run([str(a) for a in args], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stdout + r.stderr)


def f2c_all(f2c, folder):
    for c in folder.glob('*.c'):
        c.unlink()
    files = sorted(x.name for x in folder.glob('*.f'))
    r = subprocess.run([str(f2c), '-R', '-r', '-a', '-kr', '-A', *files], cwd=folder,
                       capture_output=True, text=True)
    (folder / 'f2c.log').write_text(r.stdout + r.stderr)
    if r.returncode:
        raise SystemExit(r.stdout + r.stderr)


def tidy(path):
    """Generated text ends in exactly one newline, as committed."""
    path.write_text(path.read_text().rstrip('\n') + '\n')


def struct_bytes(text, name):
    body = re.search(r'typedef struct \{([^{}]*)\} ' + name + ';', text)[1]
    total = 0
    for kind, field in re.findall(r'(real|integer|doublereal|logical)\s+([^;]+);', body):
        width = 8 if kind == 'doublereal' else 4
        for entity in field.split(','):
            m = re.search(r'\[\s*(\d+)\s*\]', entity)
            total = (total + width - 1) // width * width + width * (int(m[1]) if m else 1)
    return total


def co2_launch_unit(text):
    """The CO2 launch unit's per-column storage at MAXL, every edit counted."""
    lp1 = MAXL + 1
    edits = [
        ('float full[82],half[81];', f'float full[{lp1}],half[{MAXL}];', 1),
        ('a[k][q/np+82*(q%np)]', f'a[k][q/np+{lp1}*(q%np)]', 1),
        ('high[q+6724]', f'high[q+{lp1 * lp1}]', 1), ('high[q+13448]', f'high[q+{2 * lp1 * lp1}]', 1),
        ('low[q+6724]', f'low[q+{lp1 * lp1}]', 1), ('low[q+13448]', f'low[q+{2 * lp1 * lp1}]', 1),
        ('k=i+82*j;', f'k=i+{lp1}*j;', 1),
        ('q+82*(q+1)', f'q+{lp1}*(q+1)', 1),
        ('co2o3_sgtemp[q+82]', f'co2o3_sgtemp[q+{lp1}]', 1),
        ('q*82,', f'q*{lp1},', 2),
    ]
    for old, new, count in edits:
        if text.count(old) != count:
            raise ValueError(f'CO2 launch unit: {old!r} occurs {text.count(old)} times, expected {count}')
        text = text.replace(old, new)
    if re.search(r'\b8[12]\b', text.replace('q<81', '')):
        raise ValueError('CO2 launch unit keeps an unclassified 81/82 bound')
    return text


def main():
    wrf, f2c, work, out = map(Path, sys.argv[1:5])
    kernels = out / 'woof/core/kernels'
    kernels.mkdir(parents=True, exist_ok=True)
    col, init = work / f'column{TIER or "_l81"}', work / f'init{TIER or "_l81"}'
    for folder in (col, init):
        if folder.exists():
            shutil.rmtree(folder)
    py = sys.executable
    # Column: lowering, compiler-frozen REAL parameters, f2c, assembly, ABI.
    lowered(py, HERE / 'lower_column.py', wrf, col, f2c)
    run(py, HERE / 'round_parameters.py', col)
    for x in col.glob('*-constants*'):
        x.unlink()
    f2c_all(f2c, col)
    cuh = kernels / f'gfdleta_column{TIER}.cuh'
    run(py, HERE / 'assemble_column.py', col, cuh)
    tidy(cuh)
    layout = json.loads((col / 'scratch-layout.json').read_text())
    (col / 'scratch.json').write_text(json.dumps({'scratch_bytes': layout['bytes']}) + '\n')
    run(py, HERE / 'make_column_wrapper.py', '--wrf-source', wrf / 'phys/module_ra_gfdleta.F',
        '--scratch-layout', col / 'scratch.json', '--output-root', out)
    # CO2/ozone initialization.
    lowered(py, HERE / 'lower_init.py', wrf, init, f2c)
    run(py, HERE / 'round_parameters.py', init)
    for x in init.glob('*-constants*'):
        x.unlink()
    f2c_all(f2c, init)
    icuh = kernels / f'gfdleta_co2_init{TIER}.cuh'
    run(py, HERE / 'assemble_init.py', init, icuh)
    text = icuh.read_text()
    for fn in NOINLINE:
        text, n = re.subn(r'(GD_DEVICE )(int ' + fn + r'\()', r'\1GD_NOINLINE \2', text)
        if not n:
            raise ValueError('missing ' + fn)
    icuh.write_text(NOINLINE_NOTE + text.rstrip('\n') + '\n')
    shipped = ROOT / 'woof/core/kernels/gfdleta_co2_init.cu'
    unit = shipped.read_text() if MAXL == 81 else co2_launch_unit(shipped.read_text())
    (kernels / f'gfdleta_co2_init{TIER}.cu').write_text(unit)
    text = icuh.read_text()
    init_layout = {'table_bytes': struct_bytes(text, 'GDTable'),
                   'scratch_bytes': struct_bytes(text, 'GDScratch'),
                   'state_bytes': struct_bytes(text, 'GDInitState')}
    (out / f'woof/core/gfdleta_init_layout{TIER}.json').write_text(json.dumps(init_layout, separators=(',', ':')) + '\n')
    print(MAXL, json.loads((out / f'woof/core/gfdleta_column_layout{TIER}.json').read_text())['scratch_bytes'], init_layout)


if __name__ == '__main__':
    main()
