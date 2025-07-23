#!/bin/bash

mkdir "out"
(cd out && cmake ..)
(cd out && make)
(cd out && ./plot_model_calibration > calibration.txt)
(cd out && ./filter_tuner > filter.txt)
(cd out && ./ribbon_learned_bench > bench.txt)
