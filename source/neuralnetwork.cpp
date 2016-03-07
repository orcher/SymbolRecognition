#include "neuralnetwork.hpp"
#include "neuron.hpp"

#include <iostream>
#include <iomanip> 

#define MIN_ACCEPTABLE_ERROR 0.001f

nn::NeuralNetwork::NeuralNetwork(int inputSize, const std::vector<int> &layersSizes)
{
	for (unsigned int layer = 0; layer < layersSizes.size(); layer++)
	{
		std::vector<Neuron *> neuronLayer;
		for (int neuron = 0; neuron < layersSizes[layer]; neuron++)
			neuronLayer.push_back(new Neuron((layer == 0) ? inputSize : _network[layer - 1].size()));
		_network.push_back(neuronLayer);
	}

	print();
}

void nn::NeuralNetwork::learn(const std::vector<std::vector<float> > &trainingSetsInputs, const std::vector<std::vector<float> > &trainingSetsOutputs)
{
	int iteration = 0;
	float error = 0.0f;

	// Calculate network output
	std::vector<std::vector<float> > tmp;
	for (unsigned int trainingSet = 0; trainingSet < trainingSetsInputs.size(); trainingSet++)
		tmp.push_back(generateOutput(trainingSetsInputs[trainingSet]));

	// Calculate error
	error = meanSquaredError(tmp, trainingSetsOutputs);
	std::cout << std::setw(7) << std::fixed << std::setprecision(5) << error << std::endl << std::endl;

	do
	{
		std::vector<std::vector<float> > outputs;

		// Back propagate
		for (unsigned int trainingSet = 0; trainingSet < trainingSetsInputs.size(); trainingSet++)
		{
			calculateGradients(trainingSetsOutputs[trainingSet]);
			updateWeights();
		}

		// Calculate network output
		for (unsigned int trainingSet = 0; trainingSet < trainingSetsInputs.size(); trainingSet++)
			outputs.push_back(generateOutput(trainingSetsInputs[trainingSet]));

		print();

		// Calculate error
		error = meanSquaredError(outputs, trainingSetsOutputs);
		std::cout << std::setw(7) << std::fixed << std::setprecision(5) << error << std::endl << std::endl;
		if (error <= MIN_ACCEPTABLE_ERROR)
		{
			std::cout << "FINISHED" << std::endl << std::endl;
			break;
		}

		//getchar();
	} 
	while (++iteration < 10);
}

void nn::NeuralNetwork::recognize(const std::vector<float> realCase)
{
	std::vector<float> res = generateOutput(realCase);

	print();

	std::cout << res[0];
}

std::vector<float> nn::NeuralNetwork::generateOutput(const std::vector<float> &inputs)
{
	std::vector<float> ret;

	for (unsigned int layer = 0; layer < _network.size(); layer++)
	{
		for (unsigned int neuron = 0; neuron < _network[layer].size(); neuron++)
			_network[layer][neuron]->setInputs((layer == 0) ? inputs : ret);

		ret.clear();

		for (unsigned int neuron = 0; neuron < _network[layer].size(); neuron++)
			ret.push_back(_network[layer][neuron]->genOutput());
	}

	return ret;
}

float nn::NeuralNetwork::meanSquaredError(const std::vector<std::vector<float> > &outputs, const std::vector<std::vector<float> > &expOutputs)
{
	float ret = 0.0f;

	if (outputs.size() != expOutputs.size())
	{
		std::cout << "Error: NeuralNetwork::error()" << std::endl;
		return ret;
	}

	for (unsigned int trainingSet = 0; trainingSet < outputs.size(); trainingSet++)
	{
		for (int i = 0; i < outputs[trainingSet].size(); i++)
		{
			ret += powf(expOutputs[trainingSet][i] - outputs[trainingSet][i], 2.0f);

			std::cout << powf(expOutputs[trainingSet][i] - outputs[trainingSet][i], 2.0f) << " ";
		}
	}

	ret /= outputs.size();

	return ret;
}	

float nn::NeuralNetwork::crossEntropyError(const std::vector<std::vector<float> > &outputs, const std::vector<std::vector<float> > &expOutputs)
{
	float ret = 0.0f;

	if (outputs.size() != expOutputs.size())
	{
		std::cout << "Error: NeuralNetwork::error()" << std::endl;
		return ret;
	}

	for (unsigned int trainingSet = 0; trainingSet < outputs.size(); trainingSet++)
	{
		for (int i = 0; i < outputs[trainingSet].size(); i++)
			ret -= log(outputs[trainingSet][i]) * expOutputs[trainingSet][i];
	}

	ret /= outputs.size();

	return ret;
}

void nn::NeuralNetwork::calculateGradients(const std::vector<float> &expOutputs)
{
	for (int layer = _network.size() - 1; layer >= 0; layer--)
	{
		for (int neuron = 0; neuron < _network[layer].size(); neuron++)
		{
			if (layer == _network.size() - 1)
			{
				_network[layer][neuron]->setGradient(_network[layer][neuron]->activationFunctionPrim()*(expOutputs[neuron] - _network[layer][neuron]->output()));
			}
			else
			{
				float sum = 0.0f;
				for (int i = 0; i < _network[layer + 1].size(); i++)
					sum += _network[layer + 1][i]->weightXgradient(neuron);

				_network[layer][neuron]->setGradient(_network[layer][neuron]->activationFunctionPrim() * sum);
			}
		}
	}
}

void nn::NeuralNetwork::updateWeights()
{
	for (int layer = _network.size() - 1; layer >= 0; layer--)
	{
		for (int neuron = 0; neuron < _network[layer].size(); neuron++)
		{
			_network[layer][neuron]->updateWeights();
		}
	}
}

void nn::NeuralNetwork::print()
{
	for (unsigned int layer = 0; layer < _network.size(); layer++)
	{
		for (int neuron = 0; neuron < _network[layer].size(); neuron++)
		{
			_network[layer][neuron]->print();
		}
	}
	std::cout << std::endl;
}
