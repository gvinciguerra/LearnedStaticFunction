FROM ubuntu:24.04

# Install dependencies
RUN apt-get update && apt-get install -y git openjdk-21-jdk ant ivy
RUN apt-get install -y build-essential
RUN ln -s -T /usr/share/java/ivy.jar /usr/share/ant/lib/ivy.jar

# Clone Sux4J repository
RUN git clone https://github.com/vigna/Sux4J.git
WORKDIR /Sux4J
RUN git reset --hard abce0bf

# Compile Java code
RUN sed -i -e 's/BinIO.storeObject(new GV3CompressedFunction<>(keys, TransformationStrategies.rawByteArray(), values, false, tempDir, null, codec, peeled), functionName);/new GV3CompressedFunction<>(keys, TransformationStrategies.rawByteArray(), values, false, tempDir, null, codec, peeled).dump(functionName);/g' src/it/unimi/dsi/sux4j/mph/GV3CompressedFunction.java
RUN sed -i -e 's/BinIO.storeObject(new GV4CompressedFunction<>(keys, TransformationStrategies.rawByteArray(), values, false, tempDir, null, codec), functionName);/new GV4CompressedFunction<>(keys, TransformationStrategies.rawByteArray(), values, false, tempDir, null, codec).dump(functionName);/g' src/it/unimi/dsi/sux4j/mph/GV4CompressedFunction.java
RUN ant ivy-setupjars jar

# Compile C/C++ code
COPY benchmark.cpp /Sux4J
RUN gcc -std=c99 -O3 -march=native -c c/csf.c c/csf3.c c/csf4.c c/spooky.c
RUN g++ -std=c++17 -O3 -march=native -D CSF3 benchmark.cpp csf.o csf3.o spooky.o -o csf3
RUN g++ -std=c++17 -O3 -march=native -D CSF4 benchmark.cpp csf.o csf4.o spooky.o -o csf4

COPY data_sux4j /data_sux4j
COPY run.sh /Sux4J
ENTRYPOINT ["/bin/bash", "./run.sh"]