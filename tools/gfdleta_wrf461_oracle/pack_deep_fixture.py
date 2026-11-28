"""Pack one stock-WRF ``run_column_seed`` receipt into a layout-free fixture.

    python tools/gfdleta_wrf461_oracle/pack_deep_fixture.py BUILD STEM N OUT.npz GLIBC

``BUILD/STEM-{sigma,input,context,fields}.csv`` are written by
run_column_seed.F90 against the byte-unmodified WRF module (build.sh). The
fixture keeps every column input and every ETARA output word, keyed by WRF
field name, so the test can pack them into either storage tier's ABI.  The
initialized WRF tables are kept as one SHA-256 per table over their words in
Fortran order (the CSV order), which a GPU initialization must reproduce.
``GLIBC`` is the referee libm generation the build linked (``ldd --version``);
strict numerics replay against it through ``GPUWM_WRF_GLIBC``. The build's
``wrf-commit.txt`` and ``compiler.txt`` are kept as provenance.
"""
import csv, hashlib, sys
from pathlib import Path
import numpy as np


def rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def digest(words):
    return hashlib.sha256(np.asarray(words, '<u4').tobytes()).hexdigest()


def main():
    build, stem, n, out = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    glibc = sys.argv[5]
    sigma = {}
    for r in rows(build / f'{stem}-sigma.csv'):
        sigma.setdefault(r['field'], []).append(int(r['bits'], 16))
    inputs = rows(build / f'{stem}-input.csv')
    names = sorted({r['field'] for r in inputs})
    tables = {}
    for r in rows(build / f'{stem}-context.csv'):
        tables.setdefault(r['field'], []).append(int(r['bits'], 16))
    fields = rows(build / f'{stem}-fields.csv')
    outputs = sorted({r['field'] for r in fields})
    np.savez_compressed(
        out, n=np.int32(n), referee_glibc=np.array(glibc),
        wrf_commit=np.array((build / 'wrf-commit.txt').read_text().strip()),
        compiler=np.array((build / 'compiler.txt').read_text().strip()),
        sigma_full=np.array(sigma['SFULL'][:n + 1], np.uint32),
        sigma_half=np.array(sigma['SHALF'][:n], np.uint32),
        input_names=np.array(names),
        input_field=np.array([names.index(r['field']) for r in inputs], np.uint8),
        input_column=np.array([int(r['column']) for r in inputs], np.uint8),
        input_index=np.array([int(r['index']) for r in inputs], np.uint16),
        input_bits=np.array([int(r['bits'], 16) for r in inputs], np.uint32),
        table_names=np.array(sorted(tables)),
        table_sizes=np.array([len(tables[k]) for k in sorted(tables)], np.int32),
        table_sha256=np.array([digest(tables[k]) for k in sorted(tables)]),
        output_names=np.array(outputs),
        output_field=np.array([outputs.index(r['field']) for r in fields], np.uint8),
        output_column=np.array([int(r['column']) for r in fields], np.uint8),
        output_level=np.array([int(r['level']) for r in fields], np.uint16),
        output_bits=np.array([int(r['bits'], 16) for r in fields], np.uint32))
    print(out, out.stat().st_size, len(inputs), len(tables), len(fields))


if __name__ == '__main__':
    main()
