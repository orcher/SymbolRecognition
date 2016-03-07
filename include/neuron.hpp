#ifndef NEURON_H
#define NEURON_H

#include <vector>

namespace nn
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
		Neuron(int inputSize);
		void setInputs(const std::vector<float> &inputs);
		void setGradient(float gradient);
		float weightXgradient(int neuron) const;
		float genOutput();
		float output() const;
		float activationFunctionPrim();
		void updateWeights();
		void print();

	protected:
		ActivationFunction _activFun;
		float _output;
		std::vector<float> _inputs;
		std::vector<float> _weights;
		std::vector<float> _prevWeights;
		float _gradient;

	protected:
		float activationFunction(float x) const;
		float randomFloat(float min, float max);
	};

} // NN

#endif // NEURON_H