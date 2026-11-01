# Cover every region an H2 centre could occupy with blocking spheres (all of ZIF-1 is non-percolating for H2)
import sys, numpy as np, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'/kaggle/working/mofh2')
from ase.io import read
from scipy.spatial import cKDTree
from pores import R
a=read(sys.argv[1]); cell=a.cell.array; h=0.3
n=np.maximum(8,np.ceil(a.cell.lengths()/h).astype(int))
g=np.stack(np.meshgrid(*[np.arange(k)/k for k in n],indexing='ij'),-1).reshape(-1,3)
pts=g@cell
sh=np.array([[i,j,k] for i in(-1,0,1) for j in(-1,0,1) for k in(-1,0,1)])@cell
pos=(a.get_positions()[None]+sh[:,None]).reshape(-1,3)
rad=np.tile([R.get(s,1.7) for s in a.get_chemical_symbols()],27)
t=cKDTree(pos); d,idx=t.query(pts,k=8); surf=(d-rad[idx]).min(1)
cand=np.where(surf>0.6)[0]; covered=np.zeros(len(pts),bool); spheres=[]
ptree=cKDTree(pts)
while True:
    left=cand[~covered[cand]]
    if len(left)==0: break
    i=left[np.argmax(surf[left])]; r=surf[i]+1.6
    spheres.append((g[i],r))
    # mark covered including periodic images
    for s in sh: covered[ptree.query_ball_point(pts[i]+s,r)]=True
print(len(spheres))
for f,r in spheres: print(f"{f[0]:.5f} {f[1]:.5f} {f[2]:.5f} {r:.4f}")
