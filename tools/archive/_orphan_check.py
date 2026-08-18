"""计算候选孤儿包的逆向依赖（谁依赖它），供安全清理。"""
from importlib.metadata import distributions, requires

all_dists = sorted(d.metadata["Name"].lower() for d in distributions()
                   if d.metadata["Name"])
cands = ['torch', 'wandb', 'polars', 'sentry-sdk', 'lightning-utilities',
         'onnx', 'shapely', 'pyclipper', 'omegaconf', 'ml_dtypes', 'hf-xet',
         'networkx', 'sympy', 'mpmath', 'Jinja2', 'ninja', 'filelock',
         'fsspec', 'antlr4-python3-runtime', 'lief', 'pefile', 'altgraph',
         'pyinstaller-hooks-contrib', 'aiohappyeyeballs', 'aiosignal',
         'anyio', 'httpcore', 'h11', 'yarl', 'propcache', 'multidict',
         'frozenlist', 'PyYAML', 'regex', 'tqdm', 'requests', 'certifi',
         'urllib3', 'idna', 'charset-normalizer']
for c in cands:
    norm = c.lower().replace('-', '_')
    users = []
    for d in all_dists:
        try:
            reqs = requires(d) or []
        except Exception:
            continue
        for r in reqs:
            base = (r.split(';')[0].split('>=')[0].split('<')[0]
                    .split('==')[0].split('[')[0].strip()
                    .lower().replace('-', '_'))
            if base == norm:
                users.append(d)
    print(f"{c:<28} required-by: {users if users else '(ORPHAN)'}")