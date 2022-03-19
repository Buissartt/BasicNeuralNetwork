#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <math.h>
#include <stdlib.h>
#include <cmath>

class Functions
{
    public:
        Functions();
        enum ActivationFunction {sigmoid, tanh};
        ActivationFunction type;
        double getActivation();
        virtual ~Functions();

    protected:
        double Sigmoid(double x);
        float Sigmoid(float x);
        long double Sigmoid(long double x);

        double Tanh (double x);
        float Tanh (float x);
        long double Tanh (long double x);

    private:

};

#endif // FUNCTIONS_H
