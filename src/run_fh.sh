#!/bin/bash
# hMOF-matched GCMC: UFF framework + single-site H2 (36.7 K, 2.958 A) with Feynman-Hibbs, no charges, 12.8 A cutoff, 77 K
# usage: run_fh.sh NAME PRESSURE_BAR CYCLES INIT "a b c" OUTDIR [BLOCKFILE]
S=${WORK:-/kaggle/working/mofh2}
export RASPA_DIR=$S/raspa
NAME=$1; P=$2; CYC=$3; INIT=$4; CELLS=$5; OUT=$6; BLOCK=$7
mkdir -p "$OUT" && cd "$OUT"
cat > simulation.input <<IN
SimulationType                MonteCarlo
NumberOfCycles                $CYC
NumberOfInitializationCycles  $INIT
PrintEvery                    250
RemoveAtomNumberCodeFromLabel yes
Forcefield                    ${FF:-UFF_FH}
CutOffVDW                     12.8
ChargeMethod                  None
UseChargesFromCIFFile         no
Framework 0
FrameworkName                 $NAME
UnitCells                     $CELLS
UseTabularGrid                ${GRID:-no}
SpacingVDWGrid                0.15
NumberOfGrids                 1
GridTypes                     H_com
ExternalTemperature           ${TEMP:-77.0}
ExternalPressure              $(python3 -c "print($P*1e5)")
Movies                        yes
WriteMoviesEvery              $(( (CYC+INIT) / 40 ))
Component 0 MoleculeName      H2single
            StartingBead             0
            MoleculeDefinition       local
            IdealGasRosenbluthWeight 1.0
            TranslationProbability   1.0
            SwapProbability          1.0
            CreateNumberOfMolecules  0
$( [ -n "$BLOCK" ] && printf '            BlockPockets yes\n            BlockPocketsFileName %s\n' "$BLOCK" )
IN
$RASPA_DIR/bin/simulate simulation.input > log.txt 2>&1
