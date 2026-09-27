#!/bin/bash
# Build the patched libriichi in a separate copy of Mortal, never in the production tree.
#   rust/build.sh <production Mortal dir> <build dir>
# Produces <build dir>/pylib/libriichi.so; point RIICHI_EVAL_LIBRIICHI at <build dir>/pylib.
set -euo pipefail
SRC=${1:?production Mortal dir}; OUT=${2:?build dir}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
rsync -a --delete --exclude "mortal/libriichi.so" --exclude target "$SRC/" "$OUT/Mortal/"
cd "$OUT/Mortal"
patch -p1 --forward < "$HERE/rollout.patch"
export PATH=$HOME/.cargo/bin:$PATH
nice -n 10 cargo build --release --offline -p libriichi --lib
mkdir -p "$OUT/pylib"
# copy then rename: processes that already loaded the old .so keep its inode instead of crashing
cp target/release/libriichi.so "$OUT/pylib/libriichi.so.tmp" && mv -f "$OUT/pylib/libriichi.so.tmp" "$OUT/pylib/libriichi.so"
echo "built $OUT/pylib/libriichi.so"
