#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include "Neuron.h"

class Layer
{
public:
    Layer();
    Layer(std::vector<Neuron> neurons);
    Layer(int neuronsCount, ActivationFunctionType activationFunctionType);
    virtual ~Layer();

    void InitializeWeights(int inputsCount);

    void SetInputs(std::vector<double> inputs);

    Neuron* GetNeuron(int index);
    int GetNeuronCount();


    // Each layer got pointers to access next and previous layer faster
    // without the need of searching through a vector
    Layer* GetNextLayer();
    Layer* GetPrevLayer();

    void SetNextLayer(Layer* layer);
    void SetPrevLayer(Layer* layer);

    std::vector<double> GetOutput();

    std::vector<double> inputs;

protected:

private:
    std::vector<Neuron> neurons;

    Layer* nextLayer;
    Layer* prevLayer;
};

#endif // LAYER_H
