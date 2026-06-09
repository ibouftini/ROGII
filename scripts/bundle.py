"""Concatenate src/rogii/ into a single rogii.py for Kaggle submission."""
import os

SRC = 'src/rogii'
OUT = 'notebooks/rogii.py'
HEADER = '# auto-generated bundle — do not edit\n'
SKIP = {'__pycache__'}


def collect_files(root: str) -> list[str]:
    # collect all .py files (excluding __init__.py and __pycache__)
    top_files = []
    sub_files = []
    for dirpath, dirs, fnames in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for fn in sorted(fnames):
            if fn.endswith('.py') and fn != '__init__.py':
                path = os.path.join(dirpath, fn)
                if dirpath == root:
                    top_files.append(path)
                else:
                    sub_files.append(path)
    # pipeline.py imports from all others; place it last among top-level files
    pipeline = [f for f in top_files if f.endswith('pipeline.py')]
    other = [f for f in top_files if not f.endswith('pipeline.py')]
    return other + sub_files + pipeline


def strip_relative_imports(src: str) -> str:
    lines = []
    depth = 0
    in_rogii_import = False
    for line in src.splitlines():
        stripped = line.lstrip()
        if not in_rogii_import and (
            stripped.startswith('from rogii') or stripped.startswith('import rogii')
        ):
            in_rogii_import = True
            depth = line.count('(') - line.count(')')
            lines.append(f'# {line}  # bundled')
            if depth <= 0 and not line.rstrip().endswith('\\'):
                in_rogii_import = False
        elif in_rogii_import:
            lines.append(f'# {line}  # bundled')
            depth += line.count('(') - line.count(')')
            if depth <= 0 and not line.rstrip().endswith('\\'):
                in_rogii_import = False
        else:
            lines.append(line)
    return '\n'.join(lines)


def main() -> None:
    os.makedirs('notebooks', exist_ok=True)
    parts = [HEADER]
    for path in collect_files(SRC):
        with open(path) as f:
            code = strip_relative_imports(f.read())
        parts.append(f'\n# --- {path} ---\n{code}\n')
    with open(OUT, 'w') as f:
        f.write('\n'.join(parts))
    print(f'Bundled to {OUT} ({os.path.getsize(OUT)} bytes)')


if __name__ == '__main__':
    main()
