#include "Model.h"

//{ Constructors and Destructors
Model::Model()
{
    //ctor
}

Model::~Model()
{
    //dtor
}
//}

//{ Public functions

void Model::AddLayer(int neuronsCount, ActivationFunctionType activationFunctionType)
{
    this->layers.push_back(Layer(neuronsCount,activationFunctionType));
    this->layersCount++;

    //Only if it is not the first layer
    if(this->layersCount > 1)
    {
        Layer* lastLayer = this->GetLayer(this->layersCount-1);
        Layer* penultimateLayer = this->GetLayer(this->layersCount-2);

        lastLayer->SetPrevLayer(penultimateLayer);
        penultimateLayer->SetNextLayer(lastLayer);
    }
}

void Model::Summary()
{
    std::cout << std::endl;
    std::cout << "====================SUMMARY===================" << std::endl;
    std::cout << "  --------------------" << std::endl;
    for(int i=0; i<this->layersCount; ++i)
    {
        std::cout << "  Layer " << i+1 << " :" << std::endl;
        std::cout << "      Neurons count : " << this->GetLayer(i)->GetNeuronCount() << std::endl;
        std::cout << "      Activation function : " << this->GetLayer(i)->GetNeuron(0)->GetActivationFunction().ToString() << std::endl;
        std::cout << "  --------------------" << std::endl;
    }
    std::cout << "==============================================" << std::endl;
    std::cout << std::endl;
}

void Model::Train(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> labels, OptimizerType optimizerType, LossFunctionType lossFunctionType, int epochsCount)
{
    // Call the generic Train() method with default learning rate
    this->Train(inputs, labels, optimizerType, lossFunctionType, epochsCount, this->optimizer.GetLearningRate());
}

void Model::Train(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> labels, OptimizerType optimizerType, LossFunctionType lossFunctionType, int epochsCount, double learningRate)
{
    this->SetOptimizer(optimizerType, lossFunctionType, learningRate);

    this->Initialize(inputs.at(0).size());

    /* initialize random seed: */
    srand (time(NULL));

    for(int i=0; i<epochsCount; ++i)
    {
        int j = rand() % labels.size();
        //for(size_t j=0; j<labels.size(); ++j)
        //{
            std::vector<double> output = this->Forward(inputs.at(j));
            this->BackPropagation(labels.at(j));
        //}
        std::cout << "Epochs : " << i << std::endl;
    }
}

void Model::Debug()
{
    std::cout << std::endl;
    std::cout << "====================DEBUG====================" << std::endl;
    // For each layers
    for(int i=0; i<this->layersCount; ++i)
    {
        std::cout << "Layer : " << i+1 << std::endl;
        // For each neurons in the layer
        for(int j=0; j<this->GetLayer(i)->GetNeuronCount(); ++j)
        {
            std::cout << "  Neuron : " << j+1 << std::endl;
            std::cout << "      Bias : " << this->GetLayer(i)->GetNeuron(j)->GetBias() << std::endl;
            // For each weights in the neuron
            for(size_t k=0; k<this->GetLayer(i)->GetNeuron(j)->GetWeights().size(); ++k)
            {
                std::cout << "      Weight : " << k+1 << ", value : " << this->GetLayer(i)->GetNeuron(j)->GetWeight(k) <<std::endl;
            }
        }
    }
    std::cout << "=============================================" << std::endl;
    std::cout << std::endl;
}

//}

//{ Private functions

void Model::Initialize(int inputsCount)
{
    for(int i=0; i<this->layersCount; ++i)
    {
        Layer* currentLayer = this->GetLayer(i);
        if(i==0)
            currentLayer->InitializeWeights(inputsCount);
        else
            currentLayer->InitializeWeights(currentLayer->GetPrevLayer()->GetNeuronCount());
    }
}

void Model::SetOptimizer(OptimizerType optimizerType, LossFunctionType lossFunctionType)
{
    this->optimizer = Optimizer(optimizerType,lossFunctionType);
}

void Model::SetOptimizer(OptimizerType optimizerType, LossFunctionType lossFunctionType, double learningRate)
{
    this->optimizer = Optimizer(optimizerType,lossFunctionType,learningRate);
}

std::vector<double> Model::Forward(std::vector<double> inputs)
{
    // Get the inputs
    for(int i=0; i<this->layersCount; ++i)
    {
        Layer* currentLayer = this->GetLayer(i);
        if(i == 0)
            currentLayer->SetInputs(inputs);
        else
            //currentLayer->SetInputs(currentLayer->GetPrevLayer()->GetOutput());
            currentLayer->SetInputs(this->GetLayer(i-1)->GetOutput());
    }

    return this->GetLayer(this->layersCount - 1)->GetOutput();
}

void Model::BackPropagation(std::vector<double> yTrue)
{
    //Iterate through layers from the end
    for(int k=this->layersCount-1; k>=0; --k)
    {
        Layer* currentLayer = this->GetLayer(k);

        //Iterate through neurons of the current layer to calculate their sigmas
        for(int j=0; j<currentLayer->GetNeuronCount(); ++j)
        {
            // First we calculate sigma of the current neuron
            double oj = currentLayer->GetNeuron(j)->GetOutput();
            double sigma;
            // Output layer neuron
            if(k == this->layersCount-1)
            {
                double tj = yTrue.at(j);
                sigma = (oj - tj) * currentLayer->GetNeuron(j)->GetActivationFunction().GetDerivateActivation(oj);
            }
            // Hidden layer neuron
            else
            {
                double sum = 0;
                for(int l=0; l< this->GetLayer(k+1)->GetNeuronCount() ; ++l)
                {
                    sum += ( this->GetLayer(k+1)->GetNeuron(l)->GetWeight(j) * this->GetLayer(k+1)->GetNeuron(l)->GetSigma());
                }
                sigma = sum*currentLayer->GetNeuron(j)->GetActivationFunction().GetDerivateActivation(oj);
            }

            // Second, we set the sigma of the current neuron
            currentLayer->GetNeuron(j)->SetSigma(sigma);

            // Third, we iterate through the current neuron weights
            // to set their new values
            for(size_t i=0; i<currentLayer->GetNeuron(j)->GetWeights().size(); ++i)
            {
                double oldWeightValue = currentLayer->GetNeuron(j)->GetWeight(i);
                double lr = this->optimizer.GetLearningRate();
                double oi = currentLayer->inputs.at(i);
                double sigmaj = currentLayer->GetNeuron(j)->GetSigma();

                double newValue = (oldWeightValue - (lr * oi * sigmaj ));
                currentLayer->GetNeuron(j)->SetWeight(i,newValue);
                //std::cout << "New/Old weight value : " << newValue << " / " << oldWeightValue << std::endl;
            }

            // Fourth, we update the current neuron's bias
            double oldBiasValue = currentLayer->GetNeuron(j)->GetBias();
            double lr = this->optimizer.GetLearningRate();

            double newBiasValue = (oldBiasValue - ((lr/5.0) * sigma));
            //std::cout << "New/Old bias value : " << newBiasValue << " / " << oldBiasValue << std::endl;
            currentLayer->GetNeuron(j)->SetBias(newBiasValue);
        }
    }
}

Layer* Model::GetLayer(int index)
{
    return &this->layers.at(index);
}
//}

