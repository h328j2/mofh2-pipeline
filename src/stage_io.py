"""Move results between Kaggle notebooks.
  python stage_io.py restore [--reference]   unpack outputs attached from earlier notebooks (and, with --reference, the packaged reference runs) into $WORK
  python stage_io.py save NAME               pack this notebook's new run outputs (.data + simulation.input) and CSVs into /kaggle/working/NAME.tgz
  python stage_io.py forget QUEUE            delete run folders listed in a queue (to force recomputation after restoring reference runs)"""
import os, sys, glob, tarfile, shutil
W = os.environ.get("WORK", "/kaggle/working/mofh2")
cmd = sys.argv[1]
if cmd == "restore":
    tars = [t for t in glob.glob("/kaggle/input/**/*.tgz", recursive=True) if "reference_runs" not in t]
    if "--reference" in sys.argv: tars = glob.glob("/kaggle/input/**/reference_runs.tgz", recursive=True) + tars
    for t in tars:
        with tarfile.open(t) as tf: tf.extractall(W)
        print("restored", t)
    if not tars: print("nothing to restore (first notebook, or no earlier outputs attached)")
elif cmd == "save":
    out = f"/kaggle/working/{sys.argv[2]}.tgz"
    with tarfile.open(out, "w:gz") as tf:
        for f in glob.glob(f"{W}/runs*/*/Output/System_0/*.data") + glob.glob(f"{W}/runs*/*/simulation.input") + glob.glob(f"{W}/*.csv"):
            tf.add(f, arcname=os.path.relpath(f, W))
    print("saved", out, round(os.path.getsize(out) / 1e6, 1), "MB")
elif cmd == "forget":
    n = 0
    for l in open(sys.argv[2]):
        l = l.strip()
        if l:
            d = f"{W}/{l.split('|')[6]}"
            if os.path.isdir(d): shutil.rmtree(d); n += 1
    print("removed", n, "run folders listed in", sys.argv[2])
