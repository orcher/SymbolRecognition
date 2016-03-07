#include "neuralnetwork.hpp"

#include <vector>

void main()
{
	std::vector<std::vector<float> > trainingSetsInputs = {
		{ 1, 4 },
		{ 1, 5 },
		{ 2, 1 },
		{ 2, 5 },
		{ 3, 3 },
		{ 4, 2 } 
	};
	std::vector<std::vector<float> > trainingSetsOutputs = {
		{ 1 },
		{ 1 },
		{ 0 },
		{ 1 },
		{ 0 },
		{ 0 }
	};

	std::vector<float> realCase = { 2, 2 };

	std::vector<int> layersSizes = { 2, 1 };
	nn::NeuralNetwork net(2, layersSizes);
	net.learn(trainingSetsInputs, trainingSetsOutputs);
	net.recognize(realCase);

	getchar();
}