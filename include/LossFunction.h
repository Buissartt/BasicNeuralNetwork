#ifndef LOSSFUNCTION_H
#define LOSSFUNCTION_H

#include <math.h>
#include <vector>

enum LossFunctionType {
    MEAN_SQUARED,
    CROSS_ENTROPY
};

class LossFunction
{
    public:
        LossFunction();
        LossFunction(LossFunctionType lossFunctionType);
        virtual ~LossFunction();

        double GetLoss(std::vector<double> yTrue, std::vector<double> yPrediction);
        void SetLossFunctionType(LossFunctionType lossFunctionType);

    protected:

    private:
        LossFunctionType type;
};

#endif // LOSSFUNCTION_H
