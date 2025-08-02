# Learned Static Functions

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Static function data structures associate a static set of keys with values while allowing arbitrary output values for queries involving keys outside the set.
This enables them to use significantly less memory.
Several techniques are known, with *compressed static functions* approaching the zero-order empirical entropy of the value sequence.
*Learned static functions* use machine learning to capture correlations between keys and values.
For each key, a model predicts a probability distribution over the values, from which we derive a key-specific prefix code to compactly encode the true value.
The resulting codeword is stored in a classic static function data structure.
This design allows learned static functions to break the zero-order entropy barrier while still supporting point queries.

In this repository, we give the first implementation of a learned static function based on a [modified version of BuRR](https://github.com/stefanfred/BuRR/tree/falsepositiveVLR) that we also provide.
You can find the original here: [Code](https://github.com/lorenzhs/BuRR), [paper](https://drops.dagstuhl.de/storage/00lipics/lipics-vol233-sea2022/LIPIcs.SEA.2022.4/LIPIcs.SEA.2022.4.pdf).

### File structure

- [`train`](https://github.com/gvinciguerra/LearnedStaticFunction/tree/main/train) folder: Code for training the ML model using [TensorFlow](https://github.com/tensorflow/tensorflow/)
- [BuRR-VLR](https://github.com/stefanfred/BuRR/tree/falsepositiveVLR) repository: Implementation of our variable-length BuRR adaption
- [`include`](https://github.com/gvinciguerra/LearnedStaticFunction/tree/main/include/lsf) folder: Implementation of our learned data structure
- [`csf`](https://github.com/gvinciguerra/LearnedStaticFunction/tree/main/csf) folder: Benchmark code for the GOV competitor

### Reproducibility

For convenience, we provide a Docker image that can be used to reproduce our experiments.
To run it, clone our repo and run Docker (as superuser).

```bash
git clone --recursive https://github.com/gvinciguerra/LearnedStaticFunction.git
docker build --pull --rm -t lsf .
docker run -it -v $(pwd)/lrdata:/lrdata -v $(pwd)/data_sux4j:/data_sux4j -v $(pwd)/out:/out lsf
```

It will run the training and the benchmarks for all competitors.
It will automatically generate the paper based on those results.
The paper can be found in ```/out/main.pdf``` together with other raw benchmark outputs.
Note that the runtime is several hours.

### License

This code is licensed under the [GPLv3](/LICENSE).
