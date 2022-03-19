#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "LossFunction.h"

enum OptimizerType
{
    GRADIENT_DESCENT
};


class Optimizer
{
public:
    Optimizer();
    Optimizer(OptimizerType optimizerType, LossFunctionType lossFunctionType);
    Optimizer(OptimizerType optimizerType, LossFunctionType lossFunctionType, double learningRate);
    virtual ~Optimizer();

    double GetNewWeight(double oldWeight, double lossDerivateValue);
    void SetOptimizerType(OptimizerType type);

    double GetLearningRate();
    void SetLearningRate(double lr);

protected:

private:
    OptimizerType type;
    LossFunction lossFunction;
    double learningRate;
};

#endif // OPTIMIZER_H
