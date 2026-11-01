import glob, re, json, sys, os, numpy as np, base64
S=os.path.dirname(os.path.abspath(__file__))
def parse_data(f):
    t=open(f).read()
    def g(pat):
        m=re.search(pat,t); return float(m.group(1)) if m else None
    blk=t.split('Average loading absolute [molecules/unit cell]')
    res=dict(
      mgg=g(r'Average loading absolute \[milligram/gram framework\]\s+([-\d.eE+]+)'),
      mgg_err=g(r'Average loading absolute \[milligram/gram framework\]\s+[-\d.eE+]+\s+\+/-\s+([-\d.eE+]+)'),
      ex_mgg=g(r'Average loading excess \[milligram/gram framework\]\s+([-\d.eE+]+)'),
      n_cell=g(r'Average loading absolute \[molecules/unit cell\]\s+([-\d.eE+]+)'),
      dens=g(r'Framework Density:\s+([-\d.eE+]+) \[kg/m\^3\]'),
      cycles=g(r'Number of cycles:\s+(\d+)') ,
      finished='Simulation finished' in t)
    if res['mgg'] is not None and res['dens']: 
        res['gL']=res['mgg']*res['dens']/1000; res['gL_err']=res['mgg_err']*res['dens']/1000
    return res
def pdb_frames(f):
    frames=[];cur=None;cell=None
    for line in open(f):
        if line.startswith('MODEL'): cur=[]; frames.append(cur)
        elif line.startswith('CRYST1'): cell=[float(x) for x in line.split()[1:7]]
        elif line.startswith('ATOM') and cur is not None:
            cur.append([float(line[30:38]),float(line[38:46]),float(line[46:54])])
    return frames,cell
def framework(f):
    at=[];cell=None
    for line in open(f):
        if line.startswith('CRYST1'): cell=[float(x) for x in line.split()[1:7]]
        if line.startswith(('ATOM','HETATM')):
            el=line[76:78].strip() or re.sub(r'[^A-Za-z]','',line[12:16])[:2]
            at.append([el,float(line[30:38]),float(line[38:46]),float(line[46:54])])
    return at,cell
if __name__=='__main__':
    for d in sorted(glob.glob(S+'/runs/*_*')):
        if 'test' in d: continue
        fs=glob.glob(d+'/Output/System_0/*.data')
        if fs: print(os.path.basename(d), parse_data(fs[0]))
