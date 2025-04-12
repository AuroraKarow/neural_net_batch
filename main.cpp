#pragma once

#include "dataset"
#include "neunet"

using namespace std;
using namespace mtx;
using namespace dataset;
using namespace neunet;
using namespace layer;

int main(int argc, char *argv[], char *envp[])
{
    cout << "hello, world." << endl;
    // MNIST demo
    string root_dir = "E:\\VS Code project data\\MNIST\\";
    MNIST train_set(root_dir + "train-images.idx3-ubyte", root_dir + "train-labels.idx1-ubyte", true);
    // train_set.output_bitmap("E:\\VS Code project data\\MNIST_out\\train", BMIO_BMP);
    MNIST test_set(root_dir + "t10k-images.idx3-ubyte", root_dir + "t10k-labels.idx1-ubyte", true);
    // test_set.output_bitmap("E:\\VS Code project data\\MNIST_out\\test", BMIO_BMP);
    auto dLearnRate = 0.4;
    NetMNISTIm2Col LeNet(0.1, 125, true);
    LeNet.AddLayer<LAYER_CONV_IM2COL>(20, 5, 5, 1, 1, dLearnRate);
    LeNet.AddLayer<LAYER_CONV_BN_IM2COL>();
    LeNet.AddLayer<LAYER_ACT>(RELU);
    LeNet.AddLayer<LAYER_POOL_IM2COL>(POOL_MAX_IM2COL, 2, 2, 2, 2);
    LeNet.AddLayer<LAYER_CONV_IM2COL>(50, 5, 5, 1, 1, dLearnRate);
    LeNet.AddLayer<LAYER_CONV_BN_IM2COL>();
    LeNet.AddLayer<LAYER_ACT>(RELU);
    LeNet.AddLayer<LAYER_POOL_IM2COL>(POOL_MAX_IM2COL, 2, 2, 2, 2);
    LeNet.AddLayer<LAYER_TRANS_IM2COL>();
    LeNet.AddLayer<LAYER_FC>(500, dLearnRate);
    LeNet.AddLayer<LAYER_FC_BN>();
    LeNet.AddLayer<LAYER_ACT>(SIGMOID);
    LeNet.AddLayer<LAYER_FC>(10, dLearnRate);
    LeNet.AddLayer<LAYER_ACT>(SOFTMAX);
    cout << "[LeNet depth][" << LeNet.Depth() << ']' << endl;
    if(LeNet.Run(train_set, test_set)) return EXIT_SUCCESS;
    else return EXIT_FAILURE;
}