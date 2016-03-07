#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>

namespace nn
{

	class Neuron;

class NeuralNetwork
{

public:
	NeuralNetwork(int inputSize, const std::vector<int> &layersSizes);
	void learn(const std::vector<std::vector<float> > &inputs, const std::vector<std::vector<float> > &outputs);
	void recognize(const std::vector<float> realCase);

protected:
	std::vector<std::vector<Neuron*> > _network;

	std::vector<std::vector<float> > _gradients;
	
	std::vector<float> generateOutput(const std::vector<float> &inputs);
	float meanSquaredError(const std::vector<std::vector<float> > &outputs, const std::vector<std::vector<float> > &expOutputs);
	float crossEntropyError(const std::vector<std::vector<float> > &outputs, const std::vector<std::vector<float> > &expOutputs);
	void calculateGradients(const std::vector<float> &expOutputs);
	void updateWeights();
	void print();
};

} // NN

#endif // NEURALNETWORK_H
