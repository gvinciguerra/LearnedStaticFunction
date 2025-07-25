FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    unzip \
    zip \
    llvm-17 \
    clang-17 \
    libc++-dev \
    libc++abi-dev \
    libtbb-dev \
    cmake \
    && apt-get clean


############################ Build TensorFlow ############################
COPY lib/tensorflow /lsf/lib/tensorflow

RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.18.0/bazelisk-linux-amd64 -O bazelisk && \
    chmod +x bazelisk && \
    mv bazelisk /usr/local/bin/bazel

ENV HERMETIC_PYTHON_VERSION=3.12
WORKDIR /lsf/lib/tensorflow
RUN printf '\n\n\n\n\n\n\n\n' | ./configure
RUN bazel build //tensorflow/tools/pip_package:wheel --config=opt --copt=-march=native --repo_env=USE_PYWRAP_RULES=1 --repo_env=WHEEL_NAME=tensorflow_cpu --copt=-Wno-gnu-offsetof-extensions


############################ Build CSF ############################
COPY csf /lsf/csf

RUN apt-get install -y openjdk-21-jdk ant ivy
RUN ln -s -T /usr/share/java/ivy.jar /usr/share/ant/lib/ivy.jar

RUN git clone https://github.com/vigna/Sux4J.git /lsf/csf/Sux4J
WORKDIR /lsf/csf/Sux4J
RUN git reset --hard abce0bf

# Compile Java code
RUN sed -i -e 's/BinIO.storeObject(new GV3CompressedFunction<>(keys, TransformationStrategies.rawByteArray(), values, false, tempDir, null, codec, peeled), functionName);/new GV3CompressedFunction<>(keys, TransformationStrategies.rawByteArray(), values, false, tempDir, null, codec, peeled).dump(functionName);/g' src/it/unimi/dsi/sux4j/mph/GV3CompressedFunction.java
RUN sed -i -e 's/BinIO.storeObject(new GV4CompressedFunction<>(keys, TransformationStrategies.rawByteArray(), values, false, tempDir, null, codec), functionName);/new GV4CompressedFunction<>(keys, TransformationStrategies.rawByteArray(), values, false, tempDir, null, codec).dump(functionName);/g' src/it/unimi/dsi/sux4j/mph/GV4CompressedFunction.java
RUN ant ivy-setupjars jar

# Compile C/C++ code
RUN mv /lsf/csf/benchmark.cpp .
RUN gcc -std=c99 -O3 -march=native -c c/csf.c c/csf3.c c/csf4.c c/spooky.c
RUN g++ -std=c++17 -O3 -march=native -D CSF3 benchmark.cpp csf.o csf3.o spooky.o -o csf3
RUN g++ -std=c++17 -O3 -march=native -D CSF4 benchmark.cpp csf.o csf4.o spooky.o -o csf4


############################ Build LSF ############################
COPY . /lsf
RUN sed -i 's|tensorflow==2.19.0|/lsf/lib/tensorflow/bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow_cpu-2.19.0-cp312-cp312-linux_x86_64.whl|' /lsf/train/requirements.txt
WORKDIR /lsf
RUN mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-17 -DCMAKE_CXX_COMPILER=clang++-17
RUN cd build && make -j$(nproc)


############################ Execute ############################
RUN echo '#!/bin/bash\n\
set -e\n\
cd /lsf/train && bash run_train.sh\n\
cd /lsf/csf && bash run_csf.sh\n\
cd /lsf\n\
./build/plot_model_calibration -r /lrdata/ | tee /out/calibration.txt\n\
./build/filter_tuner | tee /out/filter.txt\n\
./build/ribbon_learned_bench -r /lrdata/ | tee /out/bench.txt' > entrypoint.sh

ENTRYPOINT ["/bin/bash", "./entrypoint.sh"]
