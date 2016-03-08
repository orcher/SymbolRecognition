#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>

namespace nn
{

class NeuralNetwork
{

	enum class ActivationFunction
	{
		SIGMOID,
		TANH,
		LINEAR
	};

	class Neuron
	{
	public:
		Neuron(unsigned int inputSize, ActivationFunction activationFunction = ActivationFunction::SIGMOID);
		void setActivationFunction(ActivationFunction activationFunction);
		float generateOutput();
		float activationFunction(float x) const;
		float activationFunctionPrim(float x) const;
		void print() const;
		static float randomFloat(float min, float max);

		ActivationFunction activFun;
		std::vector<float> inputs;
		std::vector<float> weights;
		std::vector<float> prevWeights;
		float output;
	};

public:
	NeuralNetwork(const std::vector<unsigned int> &netDimms);
	void learn(const std::vector<std::vector<float> > &inputs, const std::vector<std::vector<float> > &outputs);
	void recognize(const std::vector<float> realCase);

protected:
	std::vector<std::vector<Neuron*> > _network;
	unsigned int _inputSize;
	unsigned int _outputSize;
	
	std::vector<float> generateOutput(const std::vector<float> &inputs);
	float meanSquaredError(const std::vector<std::vector<float> > &outputs, const std::vector<std::vector<float> > &expOutputs);
	float crossEntropyError(const std::vector<std::vector<float> > &outputs, const std::vector<std::vector<float> > &expOutputs);
	void backPropagate(const std::vector<float> &expOutputs);
	void print();
};

} // NN

#endif // NEURALNETWORK_H
