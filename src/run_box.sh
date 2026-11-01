#!/bin/bash
S=${WORK:-/kaggle/working/mofh2}
export RASPA_DIR=$S/raspa
FF=$1; P=$2; OUT=$S/runs_box/${FF}_$P
mkdir -p $OUT && cd $OUT
cat > simulation.input <<IN
SimulationType                MonteCarlo
NumberOfCycles                3000
NumberOfInitializationCycles  1500
PrintEvery                    1000
Forcefield                    $FF
CutOffVDW                     12.8
ChargeMethod                  None
Box 0
BoxLengths                    30 30 30
ExternalTemperature           77.0
ExternalPressure              $(python3 -c "print($P*1e5)")
Component 0 MoleculeName      H2single
            MoleculeDefinition       local
            TranslationProbability   1.0
            SwapProbability          1.0
            CreateNumberOfMolecules  0
IN
$RASPA_DIR/bin/simulate simulation.input > log.txt 2>&1
grep -m1 "Average Density" Output/System_0/*.data | head -1; grep -A1 -m1 "Average loading absolute \[molecules" Output/System_0/*.data | head -1
