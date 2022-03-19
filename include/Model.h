#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include "Layer.h"
#include "Optimizer.h"

#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>
class Model
{
public:
    Model();
    virtual ~Model();

    void Summary();

    void AddLayer(int neuronsCount, ActivationFunctionType activationFunctionType);

    void Train(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> labels, OptimizerType optimizerType, LossFunctionType lossFunctionType, int epochsCount);
    void Train(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> labels, OptimizerType optimizerType, LossFunctionType lossFunctionType, int epochsCount, double learningRate);

    void Debug();

    std::vector<double> Forward(std::vector<double> inputs);
protected:

private:
    std::vector<Layer> layers;
    Optimizer optimizer;

    void Initialize(int inputsCount);

    void SetOptimizer(OptimizerType optimizerType,LossFunctionType lossFunctionType);
    void SetOptimizer(OptimizerType optimizerType, LossFunctionType lossFunctionType, double learningRate);


    void BackPropagation(std::vector<double> yTrue);

    int layersCount = 0;
    Layer* GetLayer(int index);
};

#endif // MODEL_H
