#!/bin/bash
# resumable queue: each line KIND|NAME|CELLS|OUTDIR  with KIND in w160 (Widom 160 K), g160 (GCMC 160 K/5 bar), g77 (GCMC 77 K/100 bar)
S=${WORK:-/kaggle/working/mofh2}
job() {
  IFS="|" read K N U O <<< "$1"
  ls "$O"/Output/System_0/*.data >/dev/null 2>&1 && grep -q "Simulation finished" "$O"/Output/System_0/*.data && return
  rm -rf "$O"
  case $K in
    w160) TEMP=160.0 FF=UFF_FHm2 $S/run_widom.sh "$N" 1000 "$U" "$O" ;;
    g160) TEMP=160.0 FF=UFF_FHm2 $S/run_fh.sh "$N" 5 1000 1000 "$U" "$O" ;;
    g77)  TEMP=77.0  FF=UFF_FHm2 $S/run_fh.sh "$N" 100 500 500 "$U" "$O" ;;
    g77s) TEMP=77.0 FF=UFF_FHm2 $S/run_fh.sh "$N" 100 250 250 "$U" "$O" ;;
    g77_2s) TEMP=77.0 FF=UFF_FHm2 $S/run_fh.sh "$N" 2 300 300 "$U" "$O" ;;
    g77t) TEMP=77.0 FF=UFF_FHm2 $S/run_fh.sh "$N" 100 400 400 "$U" "$O" ;;
    g77_2) TEMP=77.0 FF=UFF_FHm2 $S/run_fh.sh "$N" 2 1000 1000 "$U" "$O" ;;
  esac
  rm -rf "$O/Movies" "$O/Restart" "$O/Snapshots" "$O/VTK"
  echo "done $K $N $(date +%H:%M)"
}
export -f job; export S
cat "$1" | xargs -P ${NPROC:-4} -I{} bash -c 'job "{}"'
echo QUEUE_END
