#include "neuralnetwork.hpp"

#include <vector>
#include <ctime>

void main()
{
	srand(time(NULL));

	std::vector<std::vector<float> > trainingSetsInputs = {
		{ 1, 3 },
		{ 2, 4 },
		{ 4, 5 },
		{ 2, 1 },
		{ 4, 1 },
		{ 5, 3 }
	};
	std::vector<std::vector<float> > trainingSetsOutputs = {
		{ 1 },
		{ 1 },
		{ 1 },
		{ 0 },
		{ 0 },
		{ 0 }
	};

	std::vector<float> realCase = { 4, 2 };

	std::vector<int> layersSizes = { 10, 5, 1 };
	nn::NeuralNetwork net(2, layersSizes);
	net.learn(trainingSetsInputs, trainingSetsOutputs);
	net.recognize(realCase);

	getchar();
}