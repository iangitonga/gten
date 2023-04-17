# gten
**gten** ia a simple transformer neural-network library to run transformer inference on CPU. It
is implemented entirely in C++ and does not require any external dependencies. It contains
modules that can be combined to create and run transformers at-least as fast as PyTorch on CPUs.
Currently, GPT-2 model is implemented. **gten's** development is inspired by PyTorch and
[GGML](https://github.com/ggerganov/ggml). In the future, I intend to add fancy features such as
multithreading, int8 quantization and FP32 inference.


## Install and Run GPT-2 using gten.
```
git clone https://github.com/iangitonga/gten.git
cd gten
g++ -O3 -ffast-math tensor.cpp modules.cpp gpt2.cpp -o gpt2
./gpt2 -p "Once upon a time"
```

If you have an Intel CPU that supports AVX, run:
```
git clone https://github.com/iangitonga/gten.git
cd gten
g++ -O3 -ffast-math -mavx tensor.cpp modules.cpp gpt2.cpp -o gpt2
./gpt2 -p "Once upon a time"
```
