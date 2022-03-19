#include "Layer.h"

Layer::Layer()
{
    //ctor
}

Layer::~Layer()
{
    //dtor
}

Layer::Layer(std::vector<Neuron> neurons)
{
    this->neurons = neurons;
}

Layer::Layer(int neuronsCount, ActivationFunctionType activationFunctionType)
{
    for(int i=0; i<neuronsCount; i++)
    {
        this->neurons.push_back(Neuron(activationFunctionType));
    }
}

void Layer::InitializeWeights(int inputCount)
{
    int neuronsCount = this->neurons.size();
    for(int i=0; i<neuronsCount; i++)
    {
        this->neurons.at(i).InitializeWeights(inputCount, neuronsCount);
    }
}

void Layer::SetInputs(std::vector<double> inputs)
{
    for(size_t i=0; i<this->neurons.size(); i++)
    {
        this->neurons.at(i).SetInput(inputs);
    }
    this->inputs = inputs;
}

Neuron* Layer::GetNeuron(int index)
{
    return &this->neurons.at(index);
}

int Layer::GetNeuronCount()
{
    return this->neurons.size();
}

Layer* Layer::GetNextLayer()
{
    return this->nextLayer;
}

Layer* Layer::GetPrevLayer()
{
    return this->prevLayer;
}

void Layer::SetNextLayer(Layer* layer)
{
    this->nextLayer = layer;
}

void Layer::SetPrevLayer(Layer* layer)
{
    this->prevLayer = layer;
}


std::vector<double> Layer::GetOutput()
{
    std::vector<double> output;

    for(size_t i=0; i<this->neurons.size(); i++)
    {
        output.push_back(this->neurons.at(i).GetOutput());
    }

    return output;
}

