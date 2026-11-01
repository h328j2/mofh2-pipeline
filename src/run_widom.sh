#!/bin/bash
# Widom insertion of single-site H2 (matched convention) at zero loading: Henry coefficient and heat of adsorption
# usage: run_widom.sh NAME CYCLES "a b c" OUTDIR     (TEMP env, default 160.0)
S=${WORK:-/kaggle/working/mofh2}
export RASPA_DIR=$S/raspa
NAME=$1; CYC=$2; CELLS=$3; OUT=$4
mkdir -p "$OUT" && cd "$OUT"
cat > simulation.input <<IN
SimulationType                MonteCarlo
NumberOfCycles                $CYC
NumberOfInitializationCycles  0
PrintEvery                    1000
RemoveAtomNumberCodeFromLabel yes
Forcefield                    ${FF:-UFF_FHm2}
CutOffVDW                     12.8
ChargeMethod                  None
UseChargesFromCIFFile         no
Framework 0
FrameworkName                 $NAME
UnitCells                     $CELLS
ExternalTemperature           ${TEMP:-160.0}
Component 0 MoleculeName      H2single
            StartingBead             0
            MoleculeDefinition       local
            IdealGasRosenbluthWeight 1.0
            WidomProbability         1.0
            CreateNumberOfMolecules  0
IN
$RASPA_DIR/bin/simulate simulation.input > log.txt 2>&1
