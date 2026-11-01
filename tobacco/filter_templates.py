import os, sys, signal, json
sys.setrecursionlimit(10000)
import networkx as nx
os.makedirs('templates_all', exist_ok=True)
import shutil
# ct2g reads from templates/, so point a symlinked folder at the database
if os.path.islink('templates') or not os.listdir('templates'):
    pass
from ciftemplate2graph import ct2g
out = {}
names = sorted(os.listdir('template_database'))
class TO(Exception): pass
def h(s,f): raise TO()
signal.signal(signal.SIGALRM, h)
os.makedirs('templates', exist_ok=True)
for i, t in enumerate(names):
    shutil.copy(os.path.join('template_database', t), os.path.join('templates', t))
    try:
        signal.alarm(10)
        nets = list(ct2g(t))
        signal.alarm(0)
        TG, start, uc, TVT, TET = nets[0][:5]
        cns = sorted({cn for cn, lab in TVT}) if all(isinstance(x, tuple) for x in TVT) else None
        out[t] = dict(nv=len(TG.nodes()), ne=len(TG.edges()), cns=[int(c) for c in cns] if cns else None,
                      nvt=len(TVT), cat=len(nets))
    except Exception as e:
        signal.alarm(0)
        out[t] = dict(err=str(e)[:60])
    os.remove(os.path.join('templates', t))
json.dump(out, open('template_stats.json', 'w'))
print(len(out))
