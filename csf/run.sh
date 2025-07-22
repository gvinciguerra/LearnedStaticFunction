#!/bin/bash

set -e

datasets=("songs" "nids" "covertype" "urls" "gaussian0" "gaussian1" "gaussian2" "gaussian3")
structures=("3" "4")

rm -f /out/*.csf*
for d in "${datasets[@]}"; do
    for s in "${structures[@]}"; do
        echo "Building $d $s"
        java -cp sux4j-5.4.1.jar:jars/runtime/* it.unimi.dsi.sux4j.mph.GV${s}CompressedFunction -b --values /data_sux4j/${d}_y.sux4j ${d}.csf${s} /data_sux4j/${d}_X.sux4j | tee ${d}.csf${s}-construct.txt
        ./csf${s} /data_sux4j/${d} ${d}.csf${s} | tee ${d}.csf${s}-query.txt
        grep "Actual bit cost" ${d}.csf${s}-construct.txt | sed 's/.*Actual/Actual/' >>${d}.csf${s}-query.txt
        mv ${d}.csf${s}* /out/
    done
done
