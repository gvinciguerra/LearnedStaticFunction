#!/bin/bash

docker build --pull --rm -f "Dockerfile" -t compressedfunction:latest "."
docker run --rm -it -v "$(pwd)/out:/out" compressedfunction:latest