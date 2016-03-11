#include "neuralnetwork.hpp"

#include <iostream>
#include <iomanip> 

#define MAX_LEARN_ITERATIONS 1000
#define MIN_ACCEPTABLE_ERROR 0.001f

#define BACKPROP_MOBILITY_FACTOR 0.0001f
#define BACKPROP_LEARNING_FACTOR 0.25f

/*
* Neuron
*/

nn::NeuralNetwork::Neuron::Neuron(unsigned int inputSize, ActivationFunction activationFunction)
{
	activFun = activationFunction;

	inputs = std::vector<float>(inputSize, 0.0f);

	for (unsigned int i = 0; i < inputSize; i++)
		weights.push_back(randomFloat(0.0f, 1.0f));
	prevWeights = weights;

	output = 0.0f;
}

float nn::NeuralNetwork::Neuron::generateOutput()
{
	float ret = 0.0f;

	for (unsigned int i = 0; i < inputs.size(); i++)
		ret += inputs[i] * weights[i];

	ret = activationFunction(ret);

	output = ret;

	return ret;
}

float nn::NeuralNetwork::Neuron::activationFunction(float x) const
{
	switch (activFun)
	{
	case ActivationFunction::SIGMOID:
		return (1.0f / (1.0f + exp(-x)));
		break;
	case ActivationFunction::TANH:
		return (1.0f - exp(-2 * x)) / (1.0f + exp(-2 * x));
		break;
	case ActivationFunction::LINEAR:
		return x;
		break;
	default:
		return 1.0f / (1.0f + exp(-x));
	}
}

float nn::NeuralNetwork::Neuron::activationFunctionPrim(float x) const
{
	switch (activFun)
	{
	case ActivationFunction::SIGMOID:
		return x * (1.0f - x);
		break;
	case ActivationFunction::TANH:
		return 1.0f - powf(x, 2.0f);
		break;
	case ActivationFunction::LINEAR:
		return 1.0f;
		break;
	default:
		return x * (1.0f - x);
	}
}

void nn::NeuralNetwork::Neuron::print() const
{
	for (unsigned int i = 0; i < inputs.size(); i++)
		std::cout << std::setw(5) << std::fixed << std::setprecision(2) << inputs[i] << " ";
	std::cout << "| ";
	for (unsigned int i = 0; i < weights.size(); i++)
		std::cout << std::setw(5) << std::fixed << std::setprecision(2) << weights[i] << " ";
	std::cout << "| ";
	std::cout << std::setw(5) << std::fixed << std::setprecision(2) << output << std::endl;
}

float nn::NeuralNetwork::Neuron::randomFloat(float min, float max)
{
	if (max < min) return 0.0f;
	float random = ((float)rand()) / (float)RAND_MAX;
	float range = max - min;
	return (random * range) + min;
}

/*
* NeuralNetwork
*/

nn::NeuralNetwork::NeuralNetwork(const std::vector<unsigned int> &netDimms)
{
	_error = "0.00";

	if (netDimms.empty())
	{
		std::cout << "Invaild network dimmensions data!" << std::endl;
		return;
	}

	for (unsigned int i = 0; i < netDimms.size(); i++)
	{
		if (netDimms[i] <= 0)
		{
			std::cout << "Invaild network dimmensions data!" << std::endl;
			return;
		}
	}

	_inputSize = netDimms[0];
	_outputSize = netDimms[netDimms.size() - 1];

	for (unsigned int layer = 1; layer < netDimms.size(); layer++)
	{
		std::vector<Neuron *> neuronLayer;
		for (unsigned int neuron = 0; neuron < netDimms[layer]; neuron++)
			neuronLayer.push_back(new Neuron((layer == 1) ? netDimms[0] : _network[layer - 2].size(), ActivationFunction::SIGMOID));
		_network.push_back(neuronLayer);
	}
}

void nn::NeuralNetwork::learn(const std::vector<std::vector<float> > &trainingSetsInputs, const std::vector<std::vector<float> > &trainingSetsOutputs)
{
	if (trainingSetsInputs.size() != trainingSetsOutputs.size())
	{
		std::cout << "Invaild training data!" << std::endl;
		return;
	}

	for (unsigned int trainingSet = 0; trainingSet < trainingSetsInputs.size(); trainingSet++)
	{
		if (trainingSetsInputs[trainingSet].size() != _inputSize || trainingSetsOutputs[trainingSet].size() != _outputSize)
		{
			std::cout << "Invaild training data!" << std::endl;
			return;
		}
	}

	int iteration = 0;
	float error = 0.0f;

	do
	{
		std::vector<std::vector<float> > outputs;

		// Back propagate
		for (unsigned int trainingSet = 0; trainingSet < trainingSetsInputs.size(); trainingSet++)
		{
			outputs.push_back(generateOutput(trainingSetsInputs[trainingSet]));
			backPropagate(trainingSetsOutputs[trainingSet]);
		}

		// Calculate error
		error = meanSquaredError(outputs, trainingSetsOutputs);
		_error = QString::number(error);
		emit errorChanged();
		std::cout << std::setw(7) << std::fixed << std::setprecision(5) << error << std::endl << std::endl;
		if (error <= MIN_ACCEPTABLE_ERROR)
		{
			std::cout << "FINISHED" << " i = " << iteration << std::endl << std::endl;
			break;
		}
		_sleep(10);
	} 
	while (++iteration < MAX_LEARN_ITERATIONS);
}

void nn::NeuralNetwork::recognize(const std::vector<float> realCase)
{
	std::vector<float> res = generateOutput(realCase);

	print();

	for (unsigned int i = 0; i < res.size(); i++)
		std::cout << std::setw(5) << std::fixed << std::setprecision(2) << res[i] << " ";
	std::cout << std::endl;
}

std::vector<float> nn::NeuralNetwork::generateOutput(const std::vector<float> &inputs)
{
	std::vector<float> ret;

	for (unsigned int layer = 0; layer < _network.size(); layer++)
	{
		for (unsigned int neuron = 0; neuron < _network[layer].size(); neuron++)
			_network[layer][neuron]->inputs = (layer == 0) ? inputs : ret;

		ret.clear();

		for (unsigned int neuron = 0; neuron < _network[layer].size(); neuron++)
			ret.push_back(_network[layer][neuron]->generateOutput());
	}

	return ret;
}

float nn::NeuralNetwork::meanSquaredError(const std::vector<std::vector<float> > &outputs, const std::vector<std::vector<float> > &expOutputs)
{
	float ret = 0.0f;

	for (unsigned int trainingSet = 0; trainingSet < outputs.size(); trainingSet++)
	{
		for (unsigned int i = 0; i < outputs[trainingSet].size(); i++)
			ret += powf(expOutputs[trainingSet][i] - outputs[trainingSet][i], 2.0f);
	}

	ret /= outputs.size();

	return ret;
}	

float nn::NeuralNetwork::crossEntropyError(const std::vector<std::vector<float> > &outputs, const std::vector<std::vector<float> > &expOutputs)
{
	float ret = 0.0f;

	for (unsigned int trainingSet = 0; trainingSet < outputs.size(); trainingSet++)
	{
		for (unsigned int i = 0; i < outputs[trainingSet].size(); i++)
			ret -= log(outputs[trainingSet][i]) * expOutputs[trainingSet][i];
	}

	ret /= outputs.size();

	return ret;
}

void nn::NeuralNetwork::backPropagate(const std::vector<float> &expOutputs)
{
	// Calculate gradients
	std::vector<std::vector<float> > gradients;
	for (unsigned int layer = 0; layer < _network.size(); layer++)
		gradients.push_back(std::vector<float>(_network[layer].size()));

	float diff; // Difference between expected output and current output

	for (int layer = _network.size() - 1; layer >= 0; layer--)
	{
		std::vector<float> tmpGradients;
		for (unsigned int neuron = 0; neuron < _network[layer].size(); neuron++)
		{
			if (layer == _network.size() - 1)
			{
				diff = expOutputs[neuron] - _network[layer][neuron]->output;
			}
			else
			{
				diff = 0.0f;
				for (unsigned int i = 0; i < _network[layer + 1].size(); i++)
					diff += _network[layer + 1][i]->weights[neuron] * gradients[layer + 1][i];
			}
			gradients[layer][neuron] = _network[layer][neuron]->activationFunctionPrim(_network[layer][neuron]->output) * diff;
		}
	}

	// Update weights
	Neuron *currentNeuron = nullptr;

	for (int layer = _network.size() - 1; layer >= 0; layer--)
	{
		for (unsigned int neuron = 0; neuron < _network[layer].size(); neuron++)
		{
			currentNeuron = _network[layer][neuron];
			for (unsigned int weight = 0; weight < currentNeuron->weights.size(); weight++)
			{
				float prevWeight = currentNeuron->prevWeights[weight];
				currentNeuron->prevWeights[weight] = currentNeuron->weights[weight];
				currentNeuron->weights[weight] += BACKPROP_MOBILITY_FACTOR * prevWeight +
					BACKPROP_LEARNING_FACTOR * gradients[layer][neuron] * currentNeuron->inputs[weight];
			}
		}
	}
}

void nn::NeuralNetwork::print()
{
	for (unsigned int layer = 0; layer < _network.size(); layer++)
	{
		for (unsigned int neuron = 0; neuron < _network[layer].size(); neuron++)
			_network[layer][neuron]->print();
	}
	std::cout << std::endl;
}
