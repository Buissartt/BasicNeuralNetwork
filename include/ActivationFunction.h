#ifndef ACTIVATIONFUNCTION_H
#define ACTIVATIONFUNCTION_H

#include <math.h>
#include <numeric>
#include <vector>
#include <iostream>

enum ActivationFunctionType {
    SIGMOID,
    TANH,
    RELU,
    LEAKY_RELU
};

class ActivationFunction
{
    public:
        ActivationFunction();
        ActivationFunction(ActivationFunctionType type);
        virtual ~ActivationFunction();

        double GetActivation(double x);
        double GetDerivateActivation(double x);
        void SetActivationFunctionType(ActivationFunctionType type);

        std::string ToString();

    protected:

    private:
        ActivationFunctionType type;

        double sigmoid(double x);
        double relu(double x);
        double leaky_relu(double x);
};

#endif // ACTIVATIONFUNCTION_H
