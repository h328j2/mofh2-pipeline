#!/bin/bash
# One-time setup per Kaggle session (~5-8 min): copies the package to $WORK, builds RASPA2 and Zeo++, installs Python deps.
# usage: source setup_env.sh        (run from a notebook with:  !bash <pkg>/setup_env.sh)
set -e
PKG=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK=${WORK:-/kaggle/working/mofh2}
mkdir -p $WORK
cp -rn $PKG/src/. $WORK/ ; cp -rn $PKG/data/. $WORK/ ; cp -rn $PKG/model_repo $WORK/ ; cp -rn $PKG/tobacco $WORK/tob_test 2>/dev/null || true
sed -i "s#__WORK__#$WORK#g" $WORK/*.txt $WORK/*.csv 2>/dev/null || true
echo "[1/4] Python packages"
pip install -q torch_geometric pymatgen CoolProp xgboost 2>&1 | tail -1 || true
echo "[2/4] RASPA2"
if [ ! -x $WORK/raspa/bin/simulate ]; then
  rm -rf /tmp/RASPA2 && git clone -q --depth 1 https://github.com/iRASPA/RASPA2 /tmp/RASPA2
  mkdir -p $WORK/raspa/bin $WORK/raspa/share/raspa
  (cd /tmp/RASPA2 && gcc -O2 -w -fcommon -Isrc src/*.c -lm -o $WORK/raspa/bin/simulate)
  cp -r /tmp/RASPA2/share/raspa/. $WORK/raspa/share/raspa/ 2>/dev/null || true
fi
mkdir -p $WORK/raspa/share/raspa/forcefield $WORK/raspa/share/raspa/molecules/local $WORK/raspa/share/raspa/structures/cif
cp -r $PKG/forcefield/. $WORK/raspa/share/raspa/forcefield/
cp $PKG/molecules/local/*.def $WORK/raspa/share/raspa/molecules/local/
cp -n $PKG/data/raspa_cif/*.cif $WORK/raspa/share/raspa/structures/cif/
echo "[3/4] Zeo++"
if [ ! -x $WORK/ext/zeopp-lsmo/zeo++/network ]; then
  mkdir -p $WORK/ext && rm -rf $WORK/ext/zeopp-lsmo && git clone -q --depth 1 https://github.com/lsmo-epfl/zeopp-lsmo $WORK/ext/zeopp-lsmo
  INC="-I/usr/include"
  if ! (apt-get install -y -q libeigen3-dev >/dev/null 2>&1 && [ -d /usr/include/eigen3/Eigen ]); then
    git clone -q --depth 1 https://github.com/eigenteam/eigen-git-mirror /tmp/eigen && mkdir -p $WORK/ext/inc && ln -sfn /tmp/eigen $WORK/ext/inc/eigen3 && INC="-I$WORK/ext/inc"
  fi
  (cd $WORK/ext/zeopp-lsmo/voro++/src && make -s -j4 >/dev/null 2>&1)
  (cd $WORK/ext/zeopp-lsmo/zeo++ && make -s -j4 CFLAGS="-g $INC" CXXFLAGS="-g $INC" >/dev/null 2>&1)
fi
echo "[4/4] checks"
[ -x $WORK/raspa/bin/simulate ] && echo "  RASPA ok" || echo "  RASPA BUILD FAILED"
[ -x $WORK/ext/zeopp-lsmo/zeo++/network ] && echo "  Zeo++ ok" || echo "  Zeo++ BUILD FAILED"
(cd $WORK && python3 build_queues.py)
echo "WORK=$WORK ready"
