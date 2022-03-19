#include "Optimizer.h"

Optimizer::Optimizer()
{
    this->type = GRADIENT_DESCENT;
    this->lossFunction = LossFunction();
    this->learningRate = 0.01;
}

Optimizer::Optimizer(OptimizerType optimizerType, LossFunctionType lossFunctionType)
{
    this->type = optimizerType;
    this->lossFunction = LossFunction(lossFunctionType);
    this->learningRate = 0.01;
}

Optimizer::Optimizer(OptimizerType optimizerType, LossFunctionType lossFunctionType, double learningRate)
{
    this->type = optimizerType;
    this->lossFunction = LossFunction(lossFunctionType);
    this->learningRate = learningRate;
}

Optimizer::~Optimizer()
{
    //dtor
}

double Optimizer::GetNewWeight(double oldWeight, double lossDerivateValue)
{
    switch(this->type)
    {
    case(GRADIENT_DESCENT):
        return oldWeight - (this->learningRate * lossDerivateValue);
    default:
        return oldWeight;
    }
}

void Optimizer::SetOptimizerType(OptimizerType type)
{
    this->type = type;
}

double Optimizer::GetLearningRate()
{
    return this->learningRate;
}

void Optimizer::SetLearningRate(double lr)
{
    this->learningRate = lr;
}
