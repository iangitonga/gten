# gten
**gten** is a minimal library to run transformer neural network FP16 inference on CPU. It
is implemented entirely in C++ and thus does not require any external dependencies.


## Install and Run GPT-2 using gten.
```
git clone https://github.com/iangitonga/gten.git
cd gten/gpt/
g++ -std=c++17 -O3 -I../ -ffast-math gpt2.cpp -o gpt2
./gpt2 -p "Once upon a time" or ./gpt2 for a chat-interface.

If you have an Intel CPU that supports AVX and f16c compile with the following
 command to achieve ~4x performance:
 
g++ -std=c++17 -O3 -I../ -ffast-math -mavx -mf16c gpt2.cpp -o gpt2
```

## Install and Run Whisper using gten.
```
git clone https://github.com/iangitonga/gten.git
cd gten/whisper/
g++ -std=c++17 -O3 -I../ -ffast-math whisper.cpp -o whisper
./whisper assets/Elon.wav

If you have an Intel CPU that supports AVX and f16c compile with:
g++ -std=c++17 -O3 -I../ -ffast-math -mavx -mf16c whisper.cpp -o whisper
```
