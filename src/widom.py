import glob, re
R = 8.314462618e-3
def widom(outdir, T=160.0):
    f = glob.glob(f"{outdir}/Output/System_0/*.data")
    if not f: return None
    t = open(f[0]).read()
    if "Simulation finished" not in t: return None
    kh = re.search(r"\[H2single\] Average Henry coefficient:\s+([-+\deE.]+)", t)
    u = re.search(r"\[H2single\] Average  <U_gh>_1-<U_h>_0:.*?\(\s*([-+\deE.]+) \+/-", t)
    if not kh or not u: return None
    return dict(kH=float(kh.group(1)), qst0=-float(u.group(1)) + R * T)   # mol/kg/Pa, kJ/mol
