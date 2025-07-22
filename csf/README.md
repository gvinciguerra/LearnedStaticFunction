# Compressed Function Benchmark

This repository contains a Dockerfile to build and evaluate Compressed Functions from [Sux4J](https://github.com/vigna/Sux4J/).

First, ensure you already ran the [train/run_train.sh](/train/run_train.sh) to generate the datasets.
Then, execute the following commands to build and run the Docker container:

```bash
docker build --pull --rm -f "Dockerfile" -t compressedfunction:latest "."
docker run --rm -it -v "$(pwd)/out:/out" compressedfunction:latest
```

The out/ folder will get populated with files named `<dataset_name>.<compressed_function_type><suffix>`, where:
- `<dataset_name>` is the dataset name
- `<compressed_function_type>` is either csf3 or csf4
- `<suffix>` can be
  - empty: raw bytes of the data structure
  - `-construct.txt`: construction log, with construction time at the end
  - `-query.txt`: contains the query time, the hash time only, and the bits/elements