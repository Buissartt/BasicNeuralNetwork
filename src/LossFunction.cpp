#include "LossFunction.h"

LossFunction::LossFunction()
{
    this->type = MEAN_SQUARED;
}

LossFunction::LossFunction(LossFunctionType lossFunctionType)
{
    this->type = lossFunctionType;
}

LossFunction::~LossFunction()
{
    //dtor
}

double LossFunction::GetLoss(std::vector<double> yTrue, std::vector<double> yPrediction)
{
    switch(this->type)
    {
    case(MEAN_SQUARED):{
        int vectSize = yTrue.size();
        double temp = 0.0;

        for(size_t i=0; i<yTrue.size(); i++)
        {
            temp += (yTrue.at(i) - yPrediction.at(i))*(yTrue.at(i) - yPrediction.at(i));
        }
        return temp / vectSize;
    }
    case(CROSS_ENTROPY):
        return 0.0;
    default:
        return 0.0;
    }
}

void LossFunction::SetLossFunctionType(LossFunctionType lossFunctionType)
{
    this->type = lossFunctionType;
}
