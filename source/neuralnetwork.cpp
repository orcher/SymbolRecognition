#include "neuralnetwork.hpp"
#include "neuron.hpp"

#include <iostream>

#define MIN_ACCEPTABLE_ERROR 0.01f

nn::NeuralNetwork::NeuralNetwork(int inputSize, const std::vector<int> &layersSizes)
{
	if (layersSizes.size() < 2)
	{
		std::cout << "Network has to have at least 2 layers (one hidden and output layer)." << std::endl;
		return;
	}

	std::cout << "Creating network..." << std::endl;
	for (unsigned int layer = 0; layer < layersSizes.size(); layer++)
	{
		std::vector<Neuron *> neuronLayer;
		for (int neuron = 0; neuron < layersSizes[layer]; neuron++)
			neuronLayer.push_back(new Neuron((layer == 0) ? inputSize : _network[layer - 1].size()));
		_network.push_back(neuronLayer);
	}
	std::cout << "Network created." << std::endl;
}

void nn::NeuralNetwork::learn(const std::vector<std::vector<float> > &trainingSetsInputs, const std::vector<std::vector<float> > &trainingSetsOutputs)
{
	float error;
	int iteration = 0;

	do
	{
		std::vector<std::vector<float> > outputs;

		// Calculate network output
		for (unsigned int trainingSet = 0; trainingSet < trainingSetsInputs.size(); trainingSet++)
			outputs.push_back(generateOutput(trainingSetsInputs[trainingSet]));

		// Calculate error
		error = meanSquaredError(outputs, trainingSetsOutputs);
		std::cout << "MeanSquaredError: " << error << std::endl;

		if (error > MIN_ACCEPTABLE_ERROR)
		{
			// Back propagate
		}
	} 
	while (++iteration < 1);
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
			ret.push_back(_network[layer][neuron]->output());
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
			ret += powf(outputs[trainingSet][i] - expOutputs[trainingSet][i], 2.0f);
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

void nn::NeuralNetwork::calculateGradients()
{
	//float sum = 0.0;

	////Calculate gradients of neurons from output layer
	//for (unsigned int i = 0; i < outputLayerNeuronVector.size(); i++)
	//{
	//	for (unsigned int j = 0; j < outputLayerNeuronVector[i]->neuronInputVector.size(); j++)
	//		sum += outputLayerNeuronVector[i]->neuronInputVector[j] *
	//		outputLayerNeuronVector[i]->neuronWaightVector[j];

	//	outputLayerNeuronGradientVector[i] = (
	//		((1 / (1 + qExp(-sum)))*(1 - (1 / (1 + qExp(-sum)))))*
	//		(nnExpOutputVector[i] - outputLayerNeuronOutputVector[i]));
	//	sum = 0.0;
	//}

	//float sum2 = 0.0;

	////Calculate gradients of neurons from hidden layer
	//for (unsigned int i = 0; i < hiddenLayerNeuronVector.size(); i++)
	//{
	//	for (unsigned int j = 0; j < hiddenLayerNeuronVector[i]->neuronInputVector.size(); j++)
	//		sum += hiddenLayerNeuronVector[i]->neuronInputVector[j] *
	//		hiddenLayerNeuronVector[i]->neuronWaightVector[j];

	//	for (unsigned int j = 0; j < outputLayerNeuronVector.size(); j++)
	//		sum2 += outputLayerNeuronGradientVector[j] * outputLayerNeuronVector[j]->neuronWaightVector[i + 1];

	//	hiddenLayerNeuronGradientVector[i] = (
	//		((1 / (1 + qExp(-sum)))*(1 - (1 / (1 + qExp(-sum)))))*
	//		(sum2));
	//	sum = 0.0;
	//	sum2 = 0.0;
	//}
}

void nn::NeuralNetwork::updateWeights()
{
	//Updating waights of neurons from output layer
	//for (unsigned int i = 0; i < outputLayerNeuronVector.size(); i++)
	//{
	//	//Update waights
	//	for (unsigned int j = 0; j < outputLayerNeuronVector[i]->neuronWaightVector.size(); j++)
	//		outputLayerNeuronVector[i]->neuronWaightVector[j] +=
	//		LEARNING_FACTOR * outputLayerNeuronGradientVector[i] * outputLayerNeuronVector[i]->neuronInputVector[j];
	//}

	////Updating waights of neurons from hidden layer
	//for (unsigned int i = 0; i < hiddenLayerNeuronVector.size(); i++)
	//{
	//	//Update waights
	//	for (unsigned int j = 0; j < hiddenLayerNeuronVector[i]->neuronWaightVector.size(); j++)
	//		hiddenLayerNeuronVector[i]->neuronWaightVector[j] +=
	//		LEARNING_FACTOR * hiddenLayerNeuronGradientVector[i] * hiddenLayerNeuronVector[i]->neuronInputVector[j];
	//}
}
