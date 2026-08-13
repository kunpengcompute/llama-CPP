# CFLAGS="-O3 -march=armv8.6-a+sve+i8mm+fp16+f32mm -fno-unroll-loops -fomit-frame-pointer -Wno-incompatible-pointer-types" CXXFLAGS="-O3 -march=armv8.6-a+sve+i8mm+fp16+f32mm -fno-unroll-loops -fomit-frame-pointer" cmake -B build-packA-1099bd .  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF 
# CFLAGS="-O3 -fno-unroll-loops -fomit-frame-pointer -Wno-incompatible-pointer-types" CXXFLAGS="-O3 -fno-unroll-loops -fomit-frame-pointer" cmake -B build-outer-5893 .  -DCMAKE_BUILD_TYPE=Debug -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF
_FLAGS="-O3 -funroll-loops"
env CFLAGS="$_FLAGS" CXXFLAGS="$_FLAGS" \
cmake -DCMAKE_BUILD_TYPE=Release \
            -B $1 -GNinja \
            -DLLAMA_CURL=OFF -DGGML_CCACHE=OFF
cmake --build $1 --config release --target ggml-cpu llama-embedding llama-bench llama-server -j 20 
