# Ejemplo simple de mlpack

[mlpack in C++ quickstart](https://github.com/mlpack/mlpack/blob/master/doc/quickstart/cpp.md)

```bash
mkdir simple-mlpack && cd simple-mlpack

vcpkg new --application
vcpkg add port mlpack
vcpkg install

cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build build

cd build

# Get the dataset and unpack it.
wget https://www.mlpack.org/datasets/covertype-small.data.csv.gz
wget https://www.mlpack.org/datasets/covertype-small.labels.csv.gz
gunzip covertype-small.data.csv.gz covertype-small.labels.csv.gz

 ./mlpack_simple

```