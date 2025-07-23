#!/bin/bash

mkdir "out"
(cd out && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..)
(cd out && make)
(cd out && ./plot_model_calibration > calibration.txt)
(cd out && ./filter_tuner > filter.txt)
(cd out && ./ribbon_learned_bench > bench.txt)
