FROM debian:latest AS build

LABEL description="Build container - simple-mlp"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    tar \
    unzip \
    zip \
    wget \
    pkg-config \
    gfortran \
    --fix-missing \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp

RUN git clone https://github.com/Microsoft/vcpkg \ 
    && cd vcpkg \
    && ./bootstrap-vcpkg.sh 

WORKDIR /tmp

RUN wget https://github.com/Kitware/CMake/releases/download/v3.30.1/cmake-3.30.1-linux-x86_64.sh \
    && chmod +x cmake-3.30.1-linux-x86_64.sh \
    && ./cmake-3.30.1-linux-x86_64.sh --prefix=/usr/local --skip-license \
    && rm cmake-3.30.1-linux-x86_64.sh

RUN ./vcpkg/vcpkg install mlpack \
    && ./vcpkg/vcpkg integrate install

COPY . /opt/simple-mlp
WORKDIR /opt/simple-mlp

RUN cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=/tmp/vcpkg/scripts/buildsystems/vcpkg.cmake \
    && cmake --build build

#RUN mkdir build \
#    && cd out \
#    && cmake .. -DCMAKE_TOOLCHAIN_FILE=/tmp/vcpkg/scripts/buildsystems/vcpkg.cmake \
#    && make

EXPOSE 8080

FROM debian:latest AS runtime

LABEL description="Run container - simple-mlp"

RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    --fix-missing \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /usr/local/mlpack

COPY --from=build /opt/simple-mlp/build/mlp_mlpack /usr/local/mlpack/mlp_mlpack

WORKDIR /usr/local/mlpack

CMD ./mlp_mlpack

EXPOSE 8080