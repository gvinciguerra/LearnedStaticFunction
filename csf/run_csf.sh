#!/bin/bash

datasets=("songs" "nids" "covertype" "urls" "gaussian0" "gaussian1" "gaussian2" "gaussian3")
structures=("3" "4")

cd Sux4J
rm -f /out/*.csf*

for s in "${structures[@]}"; do
    rm -f /out/bench${s}.txt
    for d in "${datasets[@]}"; do
        echo "Building $d $s"
        java -cp sux4j-5.4.1.jar:jars/runtime/* it.unimi.dsi.sux4j.mph.GV${s}CompressedFunction -b --values /data_sux4j/${d}_y.sux4j ${d}.csf${s} /data_sux4j/${d}_X.sux4j | tee ${d}.csf${s}-construct.txt
        ./csf${s} /data_sux4j/${d} ${d}.csf${s} | tee ${d}.csf${s}-query.txt

        input=$(grep "Elapsed.*buckets" ${d}.csf${s}-construct.txt)
        buckets=$(echo "$input" | grep -oP '\[\K[0-9,]+(?= buckets)' | tr -d ',')
        ms_per_bucket=$(echo "$input" | grep -oP '[0-9.]+(?= ms/bucket)')
        us_per_bucket=$(echo "$input" | grep -oP '[0-9.]+(?= µs/bucket)')
        total_timems=$(awk "BEGIN {printf \"%.2f\", $buckets * $ms_per_bucket}")
        total_timeus=$(awk "BEGIN {printf \"%.2f\", $buckets * $us_per_bucket / 1000.0}")
        echo -n "RESULT comp=GOV dataset_name=$d construct_ms=$total_timems$total_timeus" >> /out/bench${s}.txt



        line=$(grep "Actual bit cost" ${d}.csf${s}-construct.txt | awk '{print $NF}')
        echo -n " storage_bits=$line" >> /out/bench${s}.txt


        lineq=$(grep "Time" ${d}.csf${s}-query.txt | awk '{print $2}')
        echo -n " query_nanos=$lineq" >> /out/bench${s}.txt


        linee=$(grep "Size" ${d}.csf${s}-query.txt | awk '{print $2}')
        echo " size=$linee" >> /out/bench${s}.txt

        mv ${d}.csf${s}* /out/
    done
done