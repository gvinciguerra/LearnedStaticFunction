#!/bin/bash

set -e

rm -rf /lrdata/*

if [ ! -d "data" ]; then
    mkdir data
    curl -L -o data/data.zip https://data.d4science.org/shub/E_NVFxRDFPWSt1U2d2U0Y5US9yUnlicVk2aDhOOHZSa2ptN0UxL24wYUhEWldYT3ljZkdDdWFaOUQzcjEycXgxSw==
    unzip data/data.zip -d data/
fi

if [ ! -d "lsfvenv" ]; then
    python3 -m venv lsfvenv
fi


source lsfvenv/bin/activate
python3 -m pip install -r requirements.txt --break-system-packages
python3 train.py

rm -rf /data_sux4j/*
mv processed_data/data_sux4j/* /data_sux4j
mv processed_data/* /lrdata
mv models/* /lrdata
