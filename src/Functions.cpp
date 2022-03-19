#include "Functions.h"

Functions::Functions()
{

}

Functions::~Functions()
{

}

double Functions::Sigmoid(double x)
{
    return (1 / (1.0+exp(-x)) );
}

float Functions::Sigmoid(float x)
{
    return (1 / (1.0+exp(-x)) );
}

long double Functions::Sigmoid(long double x)
{
    return (1 / (1.0+exp(-x)) );
}

double Functions::Tanh (double x)
{
    return tanh(x);
}
float Functions::Tanh (float x)
{
    return tanh(x);
}

long double Functions::Tanh (long double x)
{
    return tanh(x);
}
