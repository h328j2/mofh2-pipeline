import os, sys, pickle, pandas as pd
from multiprocessing import Pool
S = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, S)
import predict_topo as P
P.CACHE = S + "/core_cache"
if __name__ == "__main__":
    B = pd.read_csv(S + "/core_stageB.csv")
    cifs = [f"{S}/raspa/share/raspa/structures/cif/{r}.cif" for r in B.raspa]
    with Pool(2, maxtasksperchild=10) as p: paths = list(p.imap_unordered(P.featurise, cifs))
    st = pd.Series([pickle.load(open(x, "rb"))["status"] for x in paths]).value_counts().to_dict(); print("featurised", st)
