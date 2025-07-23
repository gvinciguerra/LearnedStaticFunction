#!/bin/bash

mkdir "out"
(cd out && cmake -DCMAKE_BUILD_TYPE=Release ..)
(cd out && make)
(./out/plot_model_calibration > calibration.txt)
(./out/filter_tuner > filter.txt)
(./out/ribbon_learned_bench > bench.txt)
