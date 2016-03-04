#include "neuron.hpp"
#include <cmath>

#define NEURON_BASE_WAIGHT 0.1f

nn::Neuron::Neuron(int inputSize)
{
	_weights = std::vector<float>(inputSize, NEURON_BASE_WAIGHT);
}

void nn::Neuron::setInputs(const std::vector<float> &inputs)
{
	_inputs = inputs;
}

float nn::Neuron::output()
{
	float output = 0.0f;

	for (unsigned int i = 0; i < _inputs.size(); i++)
		output += _inputs[i] * _weights[i];

	output = activationFunction(output);

	return output;
}

float nn::Neuron::activationFunction(float x) const
{
	// Sigmoid
	return 1 / (1 + exp(-x));
}