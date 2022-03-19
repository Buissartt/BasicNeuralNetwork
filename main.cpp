#include <iostream>

#include "Model.h"
#include "ActivationFunction.h"

#include "Optimizer.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <sys/types.h>
#include <dirent.h>
#include <stdio.h>
#include <string>
#include <filesystem>

using namespace std;
using namespace cv;

vector<double> GetImagePixelsValue(string path);
int main()
{
    /* REGRESSION LINEAIRE (equation : y=3x+1) */

    vector<vector<double>> inputs = {{-1.0},{0.0},{1.0},{2.0},{3.0},{4.0}};
    vector<vector<double>> labels = {{-2.0},{1.0},{4.0},{7.0},{10.0},{13.0}};

    Model myModel;

    myModel.AddLayer(1,RELU);

    myModel.Summary();

    myModel.Train(inputs, labels, GRADIENT_DESCENT, MEAN_SQUARED, 1000);

    cout << "10 : " << myModel.Forward({10.0}).at(0) << endl;

    myModel.Debug();

    /* CLASSIFICATION  WITH NUMBERS */
    /*
    vector<vector<double>> inputs = {{-2.0,-1.0},{25.0,6.0},{17.0,4.0},{-15.0,-6.0}};
    // Weight (lb) shift by 135 : Height (in) shift by 66
    vector<vector<double>> labels = {{1.0},{0.0},{0.0},{1.0}};
    // Women = 1 ; Men = 0;
    Model myModel;

    myModel.AddLayer(2,SIGMOID);
    myModel.AddLayer(1,SIGMOID);

    myModel.Summary();

    myModel.Train(inputs, labels, GRADIENT_DESCENT, MEAN_SQUARED, 1000);

    cout << "Emily : " << myModel.Forward({-7.0,-3.0}).at(0) << endl;
    cout << "Frank : " << myModel.Forward({20.0,2.0}).at(0) << endl;

    myModel.Debug();
    */
    /* CLASSIFICATION  WITH IMAGES (MNIST DIGITS) */

    /*vector<vector<double>> inputs;
    vector<vector<double>> labels;

    string trainingPath = "/home/antoine/Images/MNIST/DOWLOADED/trainingSet/trainingSet/";
    for(int i=0; i<10; i++)
    {
        struct dirent *entry;
        DIR *dp;
        string number = to_string(i);
        string path = trainingPath+number+"/";
        dp = opendir(&path[0]);
        if (dp == NULL)
        {
            perror("opendir: Path does not exist or could not be read.");
            return -1;
        }
        while ((entry = readdir(dp)))
        {
            string imageName = entry->d_name;
            if(imageName.substr(imageName.find(".")+1) != "jpg")
                continue;
            string imageFullPath = path+entry->d_name;
            inputs.push_back(GetImagePixelsValue(imageFullPath));
            vector<double> label;
            for(int j=0; j<10; j++)
            {
                if(j==i)
                    label.push_back(1.0);
                else
                    label.push_back(0.0);
            }
            labels.push_back(label);
        }
        closedir(dp);

        cout << "Added digit :  " << i << endl;

    }
    Model myModel;

    myModel.AddLayer(1024,RELU);
    myModel.AddLayer(10,RELU);

    myModel.Summary();

    myModel.Train(inputs, labels, GRADIENT_DESCENT, MEAN_SQUARED, 100);

    //ADD TEST HERE

    //myModel.Debug();
    vector<double> test2 = myModel.Forward(GetImagePixelsValue("/home/antoine/Images/MNIST/DOWLOADED/testSet/testSet/img_1.jpg"));
    vector<double> test3 = myModel.Forward(GetImagePixelsValue("/home/antoine/Images/MNIST/DOWLOADED/testSet/testSet/img_5.jpg"));

    for(int i=0; i<10; i++)
    {
        cout << test2.at(i) << " - ";
    }
    cout << endl;
    for(int i=0; i<10; i++)
    {
        cout << test3.at(i) << " - ";
    }*/
    return 0;
}

vector<double> GetImagePixelsValue(string path)
{
    string image_path = samples::findFile(path);
    Mat img = imread(image_path, IMREAD_GRAYSCALE);
    vector<double> pixels;

    for(int i=0; i<img.rows; i++)
    {
        for(int j=0; j<img.cols; j++)
        {
            pixels.push_back(((int)img.at<uchar>(i,j))/255.0);
        }
    }
    return pixels;
}
