#include "neuron.hpp"
#include <cmath>
#include <iostream>
#include <iomanip> 

#define MOBILITY_FACTOR 0.0001f
#define LEARNING_FACTOR 0.25f

nn::Neuron::Neuron(int inputSize)
{
	_inputs = std::vector<float>(inputSize, 0.0f);
	_weights = std::vector<float>(inputSize, randomFloat(0.0f, 1.0f));
	_prevWeights = _weights;

	_activFun = ActivationFunction::SIGMOID;
	_output = 0.0f;
	_gradient = 0.0f;
}

void nn::Neuron::setInputs(const std::vector<float> &inputs)
{
	_inputs = inputs;
}

void nn::Neuron::setGradient(float gradient)
{
	_gradient = gradient;
}

float nn::Neuron::weightXgradient(int neuron) const
{
	return _gradient * _weights[neuron];
}

float nn::Neuron::genOutput()
{
	float output = 0.0f;

	for (unsigned int i = 0; i < _inputs.size(); i++)
		output += _inputs[i] * _weights[i];

	output = activationFunction(output);

	_output = output;

	return output;
}

float nn::Neuron::output() const
{
	return _output;
}

float nn::Neuron::activationFunctionPrim()
{
	switch (_activFun)
	{
	case ActivationFunction::SIGMOID:
		return _output * (1.0f - _output);
		break;
	case ActivationFunction::TANH:
		return 1 - powf(_output, 2.0f);
		break;
	case ActivationFunction::LINEAR:
		return 1;
		break;
	default:
		return _output * (1.0f - _output);
	}
}

void nn::Neuron::updateWeights()
{
	for (int weight = 0; weight < _weights.size(); weight++)
	{
		float tmp = _prevWeights[weight];
		_prevWeights[weight] = _weights[weight];
		_weights[weight] -= MOBILITY_FACTOR * tmp + LEARNING_FACTOR * _gradient * _inputs[weight];
	}
}

void nn::Neuron::print()
{
	for (int i = 0; i < _inputs.size(); i++)
		std::cout << std::setw(5) << std::fixed << std::setprecision(2) << _inputs[i] << " ";
	std::cout << "| ";
	for (int i = 0; i < _weights.size(); i++)
		std::cout << std::setw(5) << std::fixed << std::setprecision(2) << _weights[i] << " ";
	std::cout << "| ";
	/*for (int i = 0; i < _prevWeights.size(); i++)
		std::cout << std::setw(5) << std::fixed << std::setprecision(2) << _prevWeights[i] << " ";
	std::cout << "| ";*/
	std::cout << std::setw(5) << std::fixed << std::setprecision(2) << _gradient << " ";
	std::cout << "| ";
	std::cout << std::setw(5) << std::fixed << std::setprecision(2) << _output << std::endl;
}

float nn::Neuron::activationFunction(float x) const
{
	switch (_activFun)
	{
	case ActivationFunction::SIGMOID: 
		return (1.0f / (1.0f + exp(-x)));
		break;
	case ActivationFunction::TANH:
		return (1.0f - exp(-2*x)) / (1.0f + exp(-2*x));
		break;
	case ActivationFunction::LINEAR:
		return x;
		break;
	default: 
		return 1.0f / (1.0f + exp(-x));
	}
}

float nn::Neuron::randomFloat(float min, float max)
{
	if (max < min) return 0.0f;
	float random = ((float)rand()) / (float)RAND_MAX;
	float range = max - min;
	return (random*range) + min;
}