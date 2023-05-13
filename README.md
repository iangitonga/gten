# gten
**gten** is a minimal library to run transformer neural network inference on CPU. It
is implemented entirely in C++ and does not require any external dependencies. It contains
modules that can be combined to create and run transformers as fast as PyTorch on CPUs and
supports inference in FP16(default), FP32 and Qint8(weights-only).**gten's** development is
inspired by PyTorch and [ggml](https://github.com/ggerganov/ggml).


## Install and Run GPT-2 using gten.
```
git clone https://github.com/iangitonga/gten.git
cd gten/
g++ -std=c++17 -O3 -ffast-math tensor.cpp modules.cpp gpt2.cpp -o gpt2
./gpt2 -p "Once upon a time"
```

If you have an Intel CPU that supports AVX, run:
```
git clone https://github.com/iangitonga/gten.git
cd gten/
g++ -std=c++17 -O3 -ffast-math -mavx -mf16c tensor.cpp modules.cpp gpt2.cpp -o gpt2
./gpt2 -p "Once upon a time"
```
