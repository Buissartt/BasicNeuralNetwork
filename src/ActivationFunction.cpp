#include "ActivationFunction.h"

ActivationFunction::ActivationFunction()
{
    this->type = RELU;
}

ActivationFunction::ActivationFunction(ActivationFunctionType type)
{
    this->type = type;
}

ActivationFunction::~ActivationFunction()
{
    //dtor
}

double ActivationFunction::GetActivation(double x)
{
    switch(this->type)
    {
    case(SIGMOID):
        return sigmoid(x);
    case(TANH):
        return tanh(x);
    case(RELU):
        return relu(x);
    case(LEAKY_RELU):
        return leaky_relu(x);
    default:
        return x;
    }
}

double ActivationFunction::GetDerivateActivation(double x)
{
    switch(this->type)
    {
    case(SIGMOID):
        return sigmoid(x)*(1.0-sigmoid(x));
    case(TANH):
        return 1.0-(tanh(x)*tanh(x));
    case(RELU):
        return (x > 0) ? 1.0 : 0.0;
    case(LEAKY_RELU):
        return (x > 0) ? 1.0 : 0.01;
    default:
        return x;
    }
}

void ActivationFunction::SetActivationFunctionType(ActivationFunctionType type)
{
    this->type = type;
}

double ActivationFunction::sigmoid(double x)
{
    return 0.5 + (0.5*tanh(x/2.0));
}

double ActivationFunction::relu(double x)
{
    return (x > 0) ? x : 0.0;
}

double ActivationFunction::leaky_relu(double x)
{
    return (x > 0) ? x : 0.01*x;
}

std::string ActivationFunction::ToString()
{
    switch(this->type)
    {
    case(SIGMOID):
        return "SIGMOID";
    case(TANH):
        return "TANH";
    case(RELU):
        return "RELU";
    case(LEAKY_RELU):
        return "LEAKY_RELU";
    }

    return "";
}

