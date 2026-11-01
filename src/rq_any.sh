#!/bin/bash
# Resumable parallel runner for unified queue files.
# GCMC line:  NAME|T_K|P_bar|CYCLES|FORCEFIELD|CELLS|OUTDIR(relative to $WORK)
# Widom line: NAME|T_K|W|CYCLES|FORCEFIELD|CELLS|OUTDIR
# usage: bash rq_any.sh QUEUE_FILE [MAX_HOURS]     (env: WORK, NPROC)
S=${WORK:-/kaggle/working/mofh2}; export S
export DEADLINE=$(( $(date +%s) + $(python3 -c "print(int(float('${2:-11}')*3600))") ))
job() {
  IFS="|" read N T P C F U O <<< "$1"
  O="$S/$O"
  ls "$O"/Output/System_0/*.data >/dev/null 2>&1 && grep -q "Simulation finished" "$O"/Output/System_0/*.data && return
  [ "$(date +%s)" -gt "$DEADLINE" ] && return
  rm -rf "$O"
  if [ "$P" = "W" ]; then TEMP=$T FF=$F bash $S/run_widom.sh "$N" "$C" "$U" "$O" >/dev/null 2>&1
  else TEMP=$T FF=$F bash $S/run_fh.sh "$N" "$P" "$C" "$C" "$U" "$O" >/dev/null 2>&1; fi
  rm -rf "$O/Movies" "$O/Restart" "$O/Snapshots" "$O/VTK"
  echo "done $N $T K $P $(date +%H:%M)"
}
export -f job
grep -v '^\s*$' "$1" | xargs -P ${NPROC:-4} -I{} bash -c 'job "{}"'
left=$(grep -v '^\s*$' "$1" | while IFS="|" read N T P C F U O; do ls "$S/$O"/Output/System_0/*.data >/dev/null 2>&1 && grep -q "Simulation finished" "$S/$O"/Output/System_0/*.data || echo x; done | wc -l)
echo "QUEUE_END $1 : $left runs still to do"
