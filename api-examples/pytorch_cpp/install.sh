LIBTORCH=libtorch
LIBTORCHZIP=libtorch-shared-with-deps-latest.zip
LIBTORCHURL=https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip


if [ ! -d "$LIBTORCH" ]; then
    if [ ! -f "$LIBTORCHZIP" ]; then
        wget "$LIBTORCHURL"
    fi
    unzip $LIBTORCHZIP
fi

JSON=json.hpp
JSONURL=https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp

if [ ! -f "$JSON" ]; then
    wget "$JSONURL"
fi

mkdir -p build
cd build

cmake -DCMAKE_PREFIX_PATH=$PWD/../libtorch ..
make

mv tag-text ../
mv classify-text ../
cd ..

rm -rf build
