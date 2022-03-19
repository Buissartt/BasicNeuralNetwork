#ifndef NEURON_H
#define NEURON_H

#include "ActivationFunction.h"
#include <vector>

class Neuron
{
public:
    Neuron();
    Neuron(ActivationFunctionType activationFunctionType);
    Neuron(std::vector<double> inputs, std::vector<double> weights, double bias, ActivationFunctionType activationFunction);
    virtual ~Neuron();

    void InitializeWeights(int inputsCount, int neuronsCount);

    double GetOutput();
    void SetInput(std::vector<double> inputs);

    std::vector<double> GetWeights();
    void SetWeights(std::vector<double> newWeights);

    double GetWeight(int index);
    void SetWeight(int index, double value);

    double GetBias();
    void SetBias(double newBias);

    double GetSigma();
    void SetSigma(double value);

    double Sum();

    ActivationFunction GetActivationFunction();


protected:

private:
    std::vector<double> inputs;
        std::vector<double> weights;

    double bias;
    ActivationFunction activationFunction;
    //Used by the optimizer
    double sigma;
};

#endif // NEURON_H
