#!/bin/bash
S=${WORK:-/kaggle/working/mofh2}
export RASPA_DIR=$S/raspa
NAME=$1; CELLS=$2; OUT=$S/runs/he_$1
mkdir -p "$OUT" && cd "$OUT"
cat > simulation.input <<IN
SimulationType                MonteCarlo
NumberOfCycles                2000
PrintEvery                    1000
Forcefield                    GenericMOFs
RemoveAtomNumberCodeFromLabel yes
CutOff                        12.8
Framework 0
FrameworkName                 $NAME
UnitCells                     $CELLS
ExternalTemperature           298.0
Component 0 MoleculeName      helium
            MoleculeDefinition TraPPE
            WidomProbability  1.0
            CreateNumberOfMolecules 0
IN
$RASPA_DIR/bin/simulate simulation.input > log.txt 2>&1
grep -m1 -E "\[helium\] Average Widom Rosenbluth-weight" Output/System_0/*.data
