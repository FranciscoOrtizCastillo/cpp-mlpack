# Ejemplo feedforward neural network (FNN), multilayer perceptron (MLP) con mlpack

[mlpack in C++ quickstart](https://github.com/mlpack/mlpack/blob/master/doc/quickstart/cpp.md)

```bash
mkdir simple-mlp && cd simple-mlp

vcpkg new --application
vcpkg add port mlpack
vcpkg install

cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build build

./build/mlp_mlpack

# En Windows

set "VCPKG_ROOT=D:\vcpkg"
set PATH=%VCPKG_ROOT%;%PATH%

cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake
cmake --build build
build\Debug\mlp_mlpack.exe


```

## Usando Docker

```bash
docker build -t simple-mlp/run .

docker build --target build -t simple-mlp/build .
docker build -t simple-mlp/run .

docker run -d --rm -p 8080:8080 --name simple-mlp simple-mlp/run

docker exec -it simple-mlp /bin/bash
docker stop simple-mlp
```