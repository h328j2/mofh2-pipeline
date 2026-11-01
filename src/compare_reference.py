"""Compare results reproduced on Kaggle with the original project's results (packaged copies under <pkg>/data).
usage: python compare_reference.py PKG_DIR FILE [FILE ...]"""
import sys, os, numpy as np, pandas as pd
S = os.path.dirname(os.path.abspath(__file__)); pkg = sys.argv[1]
for fn in sys.argv[2:]:
    new_p, ref_p = f"{S}/{fn}", f"{pkg}/data/{fn}"
    if not (os.path.exists(new_p) and os.path.exists(ref_p)): print(f"{fn}: missing ({'new' if not os.path.exists(new_p) else 'reference'})"); continue
    a, b = pd.read_csv(new_p), pd.read_csv(ref_p)
    num = [c for c in a.columns if c in b.columns and pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]
    n = min(len(a), len(b)); print(f"\n== {fn}: {len(a)} rows reproduced vs {len(b)} original")
    if len(a) == len(b):
        for c in num:
            d = (a[c].values[:n] - b[c].values[:n]); d = d[~np.isnan(d)]
            if len(d): print(f"   {c:28s} mean |diff| {np.abs(d).mean():9.3f}   max |diff| {np.abs(d).max():9.3f}")
    else:
        print("   row counts differ; compare by eye:"); print(a.head(8).to_string(index=False))
