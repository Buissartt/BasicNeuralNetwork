#include "Neuron.h"

#include <math.h>
#include <random>
#include <chrono>

Neuron::Neuron()
{
    //ctor
}

Neuron::~Neuron()
{
    //dtor
}

Neuron::Neuron(ActivationFunctionType activationFunctionType)
{
    this->bias = 0;
    this->activationFunction.SetActivationFunctionType(activationFunctionType);
}

Neuron::Neuron(std::vector<double> inputs, std::vector<double> weights, double bias, ActivationFunctionType activationFunctionType)
{
    this->inputs = inputs;
    this->weights = weights;
    this->bias = bias;
    activationFunction.SetActivationFunctionType(activationFunctionType);
}

/*! \brief Initialize all the current neuron weights with Xavier initialization
 *
 * \param inputsCount : the number of current neuron's input
 * \param neuronsCount : the number of neurons on the current layer
 * \return
 *
 */
void Neuron::InitializeWeights(int inputsCount, int neuronsCount)
{
    /* Create random engine with the help of seed */
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::default_random_engine e (seed);

    std::normal_distribution<double> distN(0,1);

    // rand is a random number from the normal distribution
    // coef is used by Xavier initialization
    double rand = 0.0, coef = sqrt(1.0/inputsCount);
    //double rand = 0.0, coef = sqrt(2.0/(inputsCount+neuronsCount));

    for(int j=0; j<inputsCount; j++)
    {
        rand = distN(e);
        // Do not accept zero in order to avoid dead neuron
        while(rand==0)
            rand = distN(e);

        this->weights.push_back(rand * coef);
    }
}

double Neuron::GetOutput()
{
    return this->activationFunction.GetActivation(this->Sum());
}

void Neuron::SetInput(std::vector<double> inputs)
{
    this->inputs = inputs;
}

double Neuron::Sum()
{
    double temp = 0.0;

    for(size_t i=0; i<this->inputs.size(); i++)
    {
        temp += (inputs.at(i) * weights.at(i));
    }

    return temp + this->bias;
}

std::vector<double> Neuron::GetWeights()
{
    return this->weights;
}

void Neuron::SetWeights(std::vector<double>newWeights)
{
    this->weights = newWeights;
}

double Neuron::GetWeight(int index)
{
    return this->weights.at(index);
}

void Neuron::SetWeight(int index, double value)
{
    this->weights.at(index) = value;
}

double Neuron::GetBias()
{
    return this->bias;
}

void Neuron::SetBias(double newBias)
{
    this->bias = newBias;
}

double Neuron::GetSigma()
{
    return this->sigma;
}

void Neuron::SetSigma(double value)
{
    this->sigma = value;
}

ActivationFunction Neuron::GetActivationFunction()
{
    return this->activationFunction;
}


