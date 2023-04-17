#include "tensor.h"
#include "modules.h"

#include <iostream>
#include <string_view>


const char *usage = R"(
usage: gpt2 [-m MODEL] -p PROMPT
  MODEL: Optional GPT2 model to use for inference. One of (sm, md, lg). Default is md.
  PROMPT: Required prompt to the model. Must not exceed roughly 1024 words. 

Examples:
  Prompt using default model: ./gpt2 -p "Once upon a time"
  Prompt using large model  : ./gpt2 -m lg -p "Once upon a time"
)";


int main(int argc, char const *argv[])
{
    if (!(argc == 3 || argc == 5)) {
		std::cout << "Incorrect number of arguments.\n";
		std::cout << usage << "\n";
		return -1;
	}

	const char *model_name = "GPT-2-345M";
	const char *prompt;

	if (argc == 3)
	{
		std::string_view argspec(argv[1]);
		if (argspec != "-p") {
			if (argspec == "-m") std::cerr << "Prompt not provided.\n";
			else std::cerr << "Unknown argument: " << argspec << "\n";
			std::cout << usage << "\n";
			return -1;
		}
		prompt = argv[2];
	}
	else
	{
		std::string_view model_argspec(argv[1]);
		std::string_view model_arg(argv[2]);
		std::string_view prompt_argspec(argv[3]);
		if (model_argspec != "-m" || prompt_argspec != "-p") {
			std::cout << "Wrong order of arguments or unknown arguments.\n";
			std::cout << usage << "\n";
			return -1;
		}
		prompt = argv[4];
		if (model_arg == "sm") model_name = "GPT-2-117M";
		else if (model_arg == "md") model_name = "GPT-2-345M";
		else if (model_arg == "lg") model_name = "GPT-2-762M";
		else {
			std::cout << "Unknown model: " << model_arg << "\n";
			std::cout << usage << "\n";
			return -1;
		}
	}

#if defined(__WIN32__) || defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
    std::string command = std::string("python model_dl.py ") + std::string(model_name);
#else
    std::string command = std::string("python3 model_dl.py ") + std::string(model_name);
#endif
    int res = std::system(command.c_str());
    if (res != 0) {
        std::cout << "Error: Failed to download model due to network issues.\n";
        return -1;
    }

    std::string model_dir = std::string("models/") + model_name + ".gten";
    gten::GPT2 model(model_dir);

    model.sample(prompt, 0.9);

    model.show_performance();
    return 0;
}

