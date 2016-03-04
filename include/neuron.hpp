#ifndef NEURON_H
#define NEURON_H

#include <vector>

namespace nn
{

	class Neuron
	{
	public:
		Neuron(int inputSize);
		void setInputs(const std::vector<float> &inputs);
		float output();

	protected:
		std::vector<float> _inputs;
		std::vector<float> _weights;

	protected:
		float activationFunction(float x) const;
	};

} // NN

#endif // NEURON_H