#include <iostream>
#include "pch.h"

using namespace std;
using namespace Matrix;
using namespace batnet;
using namespace dataset;
using namespace csvio;
using namespace imgdig;

int main(int argc, char *argv[], char *envp[])
{
    // Input initialization
    cout << "Train Set Initialization ... 0%";
    algo_queue<algo_queue<matrix>> batInput(32);
    algo_queue<int> batLbl(32);
    algo_queue<matrix> batOrigin(32);
    auto i = 0;
    for(auto j=0; j<9; j++)
    {
        auto imgTemp = GetImgBitMat("D:\\Users\\Aurora\\Documents\\Visual Studio Code Project\\IMAGE\\out\\0\\" + std::to_string(j) + ".bmp");
        batInput[i++] = imgTemp.mapImgCh;
        batLbl[i++] = 0;
    }
    cout << "\rTrain Set Initialization ... 33%";
    for(auto j=0; j<10; j++)
    {
        auto imgTemp = GetImgBitMat("D:\\Users\\Aurora\\Documents\\Visual Studio Code Project\\IMAGE\\out\\1\\" + std::to_string(j) + ".bmp");
        batInput[i++] = imgTemp.mapImgCh;
        batLbl[i++] = 1;
    }
    cout << "\rTrain Set Initialization ... 67%";
    for(auto j=0; j<13; j++)
    {
        auto imgTemp = GetImgBitMat("D:\\Users\\Aurora\\Documents\\Visual Studio Code Project\\IMAGE\\out\\2\\" + std::to_string(j) + ".bmp");
        batInput[i++] = imgTemp.mapImgCh;
        batLbl[i++] = 2;
    }
    for(auto i=0; i<32; i++)
    {
        matrix origin(3, 1);
        origin[batLbl[i]][0] = 1;
        batOrigin[i] = origin;
    }
    cout << "\rTrain Set Initialization ... 100%" << endl;
    cout << "Completed." << endl;

    cout << "Neural Network Initialization ... 0%";
    // Model
    // kernel_0
    auto kernel_0 = InitKernel(96, 3, 11, 11);
    // conv_BN_0
    auto gamma_conv_0 = InitScaleBatch(96), beta_conv_0 = InitShiftBatch(96);
    cout << "\rNeural Network Initialization ... 14%";
    // kernel_1
    auto kernel_1 = InitKernel(256, 96, 5, 5);
    // conv_BN_1
    auto gamma_conv_1 = InitScaleBatch(256), beta_conv_1 = InitShiftBatch(256);
    cout << "\rNeural Network Initialization ... 26%";
    // kernel_2
    auto kernel_2 = InitKernel(384, 256, 3, 3);
    // conv_BN_2
    auto gamma_conv_2 = InitScaleBatch(384), beta_conv_2 = InitShiftBatch(384);
    cout << "\rNeural Network Initialization ... 39%";
    // kernel_3
    auto kernel_3 = InitKernel(384, 384, 3, 3);
    // conv_BN_3
    auto gamma_conv_3 = InitScaleBatch(384), beta_conv_3 = InitShiftBatch(384);
    cout << "\rNeural Network Initialization ... 47%";
    // kernel_4
    auto kernel_4 = InitKernel(384, 256, 3, 3);
    // conv_BN_4
    auto gamma_conv_4 = InitScaleBatch(384), beta_conv_4 = InitShiftBatch(384);
    cout << "\rNeural Network Initialization ... 58%";
    // weight_0
    auto weight_0 = InitWeight(9216, 4096);
    // fcn_BN_0
    double gamma_fcn_0 = 1, beta_fcn_0 = 0;
    // weight_1
    auto weight_1 = InitWeight(4096, 1024);
    cout << "\rNeural Network Initialization ... 69%";
    // fcn_BN_1
    double gamma_fcn_1 = 1, beta_fcn_1 = 0;
    // weight_2
    auto weight_2 = InitWeight(1024, 512);
    cout << "\rNeural Network Initialization ... 72%";
    // fcn_BN_2
    double gamma_fcn_2 = 1, beta_fcn_2 = 0;
    // weight_3
    auto weight_3 = InitWeight(512, 64);
    cout << "\rNeural Network Initialization ... 84%";
    // fcn_BN_3
    double gamma_fcn_3 = 1, beta_fcn_3 = 0;
    cout << "\rNeural Network Initialization ... 98%";
    // weight_4
    auto weight_4 = InitWeight(64, 3);

    auto flag = true;
    double acc = 0.1;
    auto cnt = 0;
    cout << "\rNeural Network Initialization ... 100%" << endl;
    cout << "Completed." << endl;

    cout << "Iteration Starting" << endl << endl;

    while(flag)
    {
        cout << "\rConvoluted Neural Network ... 0%";
        // conv_0_sig
        auto conv_sig_0 = ConvForwProp(batInput, kernel_0, 4);
        auto conv_sig_merge_0 = MergeChannel(conv_sig_0);
        cout << "\rConvoluted Neural Network ... 8%";
        // conv_0_bn
        auto conv_bn_0 = ConvBatchNormalizationForwProp(conv_sig_merge_0, beta_conv_0, gamma_conv_0);
        auto conv_act_0 = Activate(conv_bn_0.batNormOutput, ReLU);
        cout << "\rConvoluted Neural Network ... 17%";
        // pool_0
        auto pool_0 = PoolForwProp(conv_act_0, 2, 2, false);
        // pad_0
        auto pad_0 = BatPadding(pool_0, 2, 2);
        cout << "\rConvoluted Neural Network ... 26%";
        // conv_1_sig
        auto conv_sig_1 = ConvForwProp(pad_0, kernel_1, 2);
        auto conv_sig_merge_1 = MergeChannel(conv_sig_1);
        cout << "\rConvoluted Neural Network ... 34%";
        // conv_1_bn
        auto conv_bn_1 = ConvBatchNormalizationForwProp(conv_sig_merge_1, beta_conv_1, gamma_conv_1);
        auto conv_act_1 = Activate(conv_bn_1.batNormOutput, ReLU);
        // pool_1
        auto pool_1 = PoolForwProp(conv_act_1, 2, 2, false);
        // pad_1
        auto pad_1 = BatPadding(pool_1, 1, 1);
        cout << "\rConvoluted Neural Network ... 46%";
        // conv_2_sig
        auto conv_sig_2 = ConvForwProp(pad_1, kernel_2, 2);
        auto conv_sig_merge_2 = MergeChannel(conv_sig_2);
        cout << "\rConvoluted Neural Network ... 58%";
        // conv_2_bn
        auto conv_bn_2 = ConvBatchNormalizationForwProp(conv_sig_merge_2, beta_conv_2, gamma_conv_2);
        auto conv_act_2 = Activate(conv_bn_2.batNormOutput, ReLU);
        // pad_2
        auto pad_2 = BatPadding(conv_act_2, 1, 1);
        cout << "\rConvoluted Neural Network ... 68%";
        // conv_3_sig
        auto conv_sig_3 = ConvForwProp(pad_2, kernel_3, 1);
        auto conv_sig_merge_3 = MergeChannel(conv_sig_3);
        cout << "\rConvoluted Neural Network ... 75%";
        // conv_3_bn
        auto conv_bn_3 = ConvBatchNormalizationForwProp(conv_sig_merge_3, beta_conv_3, gamma_conv_3);
        auto conv_act_3 = Activate(conv_bn_3.batNormOutput, ReLU);
        // pad_3
        auto pad_3 = BatPadding(conv_act_3, 1, 1);
        cout << "\rConvoluted Neural Network ... 84%";
        // conv_4_sig
        auto conv_sig_4 = ConvForwProp(pad_3, kernel_4, 1);
        auto conv_sig_merge_4 = MergeChannel(conv_sig_4);
        cout << "\rConvoluted Neural Network ... 91%";
        // conv_4_bn
        auto conv_bn_4 = ConvBatchNormalizationForwProp(conv_sig_merge_4, beta_conv_4, gamma_conv_4);
        auto conv_act_4 = Activate(conv_bn_4.batNormOutput, ReLU);
        // pool_2
        auto pool_2 = PoolForwProp(conv_act_4, 2, 2, false);
        cout << "\rConvoluted Neural Network ... 100%" << endl;
        cout << "Completed." << endl;
        cout << "\rFully Connected Neural Network ... 0%";
        // fcn
        auto fcn_input = CNNFCNIO(pool_2);
        // fcn_0_sig
        auto fcn_sig_0 = ForwProp(fcn_input, weight_0);
        // fcn_0_bn
        auto fcn_sig_bn_0 = BatchNormalizationForwProp(fcn_sig_0, beta_fcn_0, gamma_fcn_0);
        auto fcn_act_0 = Activate(fcn_sig_0, sigmoid);
        cout << "\rFully Connected Neural Network ... 25%";
        // fcn_1_sig
        auto fcn_sig_1 = ForwProp(fcn_act_0, weight_1);
        // fcn_1_bn
        auto fcn_sig_bn_1 = BatchNormalizationForwProp(fcn_sig_1, beta_fcn_1, gamma_fcn_1);
        auto fcn_act_1 = Activate(fcn_sig_1, sigmoid);
        cout << "\rFully Connected Neural Network ... 50%";
        // fcn_2_sig
        auto fcn_sig_2 = ForwProp(fcn_act_1, weight_2);
        // fcn_2_bn
        auto fcn_sig_bn_2 = BatchNormalizationForwProp(fcn_sig_2, beta_fcn_2, gamma_fcn_2);
        auto fcn_act_2 = Activate(fcn_sig_2, sigmoid);
        cout << "\rFully Connected Neural Network ... 75%";
        // fcn_3_sig
        auto fcn_sig_3 = ForwProp(fcn_act_2, weight_3);
        // fcn_3_bn
        auto fcn_sig_bn_3 = BatchNormalizationForwProp(fcn_sig_3, beta_fcn_3, gamma_fcn_3);
        auto fcn_act_3 = Activate(fcn_sig_bn_3.batNormOutput, sigmoid);
        // output
        auto bat_output = GaussConnForwProp(fcn_act_3, weight_4);
        cout << "\rFully Connected Neural Network ... 100%" << endl;
        cout << "Result" << endl;

        for(auto i=0; i<bat_output.size(); i++)
        {
            cout << bat_output[i] << endl;
            cout << endl;
        }
        cout << "Round " << cnt ++ << endl << endl;

        for(auto i=0; i<bat_output.size(); i++)
        {
            auto threshold = bat_output[i] - batOrigin[i];
            for(auto j=0; j<threshold.get_line(); j++)
                for(auto k=0; k<threshold.get_column(); k++)
                    if(absolute_value(threshold[j][k]) > 0.1)
                    {
                        flag = true;
                        goto bp;
                    }
            flag = false;
        }
        bp : if(flag)
        {
            cout << "Back Propagation starting" << endl;
            cout << "\rFully Connected Neural Network Back Propagation ... 0%";
            // bp
            // output
            auto bat_output_err = GaussConnPreErr(fcn_act_3, bat_output, batOrigin, weight_4, 0.01);
            // fcn_3
            auto fcn_act_3_err = DerivativeErr(bat_output_err, fcn_sig_bn_3.batNormOutput, sigmoid_derivative);
            auto fcn_sig_bn_3_err = BatchNormalizationPreErr(fcn_sig_bn_3, fcn_sig_3, fcn_act_3_err, gamma_fcn_3);
            cout << "\rFully Connected Neural Network Back Propagation ... 28%";
            auto fcn_sig_3_err = PreErr(fcn_sig_bn_3_err, weight_3);
            gamma_fcn_3 -= BNScaleGrad(fcn_act_3_err, fcn_sig_bn_3);
            beta_fcn_3 -= BNShiftGrad(fcn_act_3_err);
            weight_3 -= WeightGrad(fcn_act_3_err, fcn_act_2);
            // fcn_2
            auto fcn_act_2_err = DerivativeErr(fcn_sig_3_err, fcn_sig_bn_2.batNormOutput, sigmoid_derivative);
            auto fcn_sig_bn_2_err = BatchNormalizationPreErr(fcn_sig_bn_2, fcn_sig_2, fcn_act_2_err, gamma_fcn_2);
            auto fcn_sig_2_err = PreErr(fcn_sig_bn_2_err, weight_2);
            cout << "\rFully Connected Neural Network Back Propagation ... 50%";
            gamma_fcn_2 -= BNScaleGrad(fcn_act_2_err, fcn_sig_bn_2);
            beta_fcn_2 -= BNShiftGrad(fcn_act_2_err);
            weight_2 -= WeightGrad(fcn_act_2_err, fcn_act_1);
            // fcn_1
            auto fcn_act_1_err = DerivativeErr(fcn_sig_2_err, fcn_sig_bn_1.batNormOutput, sigmoid_derivative);
            auto fcn_sig_bn_1_err = BatchNormalizationPreErr(fcn_sig_bn_1, fcn_sig_1, fcn_act_1_err, gamma_fcn_1);
            cout << "\rFully Connected Neural Network Back Propagation ... 77%";
            auto fcn_input_err = PreErr(fcn_sig_bn_1_err, weight_1);
            gamma_fcn_1 -= BNScaleGrad(fcn_act_1_err, fcn_sig_bn_1);
            beta_fcn_1 -= BNShiftGrad(fcn_act_1_err);
            weight_1 -= WeightGrad(fcn_act_1_err, fcn_act_0);
            cout << "\rFully Connected Neural Network Back Propagation ... 100%";
            cout << "Completed" << endl;
            cout << "\rConvoluted Neural Network Back Propagation ... 0%";
            // pool_2
            auto pool_2_pre_err = CNNFCNIO(fcn_input_err, 256, 4, 4);
            auto pool_2_err = PoolPreErr(pool_2_pre_err, conv_act_4, 2, 2, false);
            // conv_4
            auto conv_act_4_err = DerivativeErr(pool_2_err, conv_bn_4.batNormOutput, ReLU_derivative);
            auto conv_bn_4_err = ConvBatchNormalizationPreErr(conv_bn_4, conv_sig_merge_4, conv_act_4_err, gamma_conv_4);
            auto gamma_conv_4_grad = ConvBNScaleGrad(conv_act_4_err, conv_bn_4);
            cout << "\rConvoluted Neural Network Back Propagation ... 13%";
            auto beta_conv_4_grad = ConvBNShiftGrad(conv_act_4_err);
            UpdateScaleShift(gamma_conv_4, gamma_conv_4_grad);
            UpdateScaleShift(beta_conv_4, beta_conv_4_grad);
            auto conv_sig_4_err = ConvPreErr(conv_bn_4_err, kernel_4, 1);
            cout << "\rConvoluted Neural Network Back Propagation ... 26%";
            // pad_3
            auto pad_3_err = BatCrop(conv_sig_4_err, 1, 1, 1, 1);
            // conv_3
            auto conv_act_3_err = DerivativeErr(pad_3_err, conv_bn_3.batNormOutput, ReLU_derivative);
            auto conv_bn_3_err = ConvBatchNormalizationPreErr(conv_bn_3, conv_sig_merge_3, conv_act_3_err, gamma_conv_3);
            auto gamma_conv_3_grad = ConvBNScaleGrad(conv_act_3_err, conv_bn_3);
            auto beta_conv_3_grad = ConvBNShiftGrad(conv_act_3_err);
            cout << "\rConvoluted Neural Network Back Propagation ... 37%";
            UpdateScaleShift(gamma_conv_3, gamma_conv_3_grad);
            UpdateScaleShift(beta_conv_3, beta_conv_3_grad);
            auto conv_sig_3_err = ConvPreErr(conv_bn_3_err, kernel_3, 1);
            // pad_2
            auto pad_2_err = BatCrop(conv_sig_3_err, 1, 1, 1, 1);
            cout << "\rConvoluted Neural Network Back Propagation ... 50%";
            // conv_2
            auto conv_act_2_err = DerivativeErr(pad_2_err, conv_bn_2.batNormOutput, ReLU_derivative);
            auto conv_bn_2_err = ConvBatchNormalizationPreErr(conv_bn_2, conv_sig_merge_2, conv_act_2_err, gamma_conv_2);
            auto gamma_conv_2_grad = ConvBNScaleGrad(conv_act_2_err, conv_bn_2);
            auto beta_conv_2_grad = ConvBNShiftGrad(conv_act_2_err);
            UpdateScaleShift(gamma_conv_2, gamma_conv_2_grad);
            UpdateScaleShift(beta_conv_2, beta_conv_2_grad);
            auto conv_sig_2_err = ConvPreErr(conv_bn_2_err, kernel_2, 2);
            cout << "\rConvoluted Neural Network Back Propagation ... 69%";
            // pad_1
            auto pad_1_err = BatCrop(conv_sig_2_err, 1, 1, 1, 1);
            // pool_1
            auto pool_1_err = PoolPreErr(pad_1_err, conv_act_1, 2, 2, false);
            // conv_1
            auto conv_act_1_err = DerivativeErr(pool_1_err, conv_bn_1.batNormOutput, ReLU_derivative);
            auto conv_bn_1_err = ConvBatchNormalizationPreErr(conv_bn_1, conv_sig_merge_1, conv_act_1_err, gamma_conv_1);
            auto gamma_conv_1_grad = ConvBNScaleGrad(conv_act_1_err, conv_bn_1);
            auto beta_conv_1_grad = ConvBNShiftGrad(conv_act_1_err);
            UpdateScaleShift(gamma_conv_1, gamma_conv_1_grad);
            UpdateScaleShift(beta_conv_1, beta_conv_1_grad);
            auto conv_sig_1_err = ConvPreErr(conv_bn_1_err, kernel_1, 2);
            cout << "\rConvoluted Neural Network Back Propagation ... 74%";
            // pad_0
            auto pad_0_err = BatCrop(pool_0, 2, 2, 2, 2);
            // pool_0
            auto pool_0_err = PoolPreErr(pad_0_err, conv_act_0, 2, 2, false);
            // conv_0
            auto conv_act_0_err = DerivativeErr(pool_0_err, conv_bn_0.batNormOutput, ReLU_derivative);
            auto conv_bn_0_err = ConvBatchNormalizationPreErr(conv_bn_0, conv_sig_merge_0, conv_act_0_err, gamma_conv_0);
            auto gamma_conv_0_grad = ConvBNScaleGrad(conv_act_0_err, conv_bn_0);
            cout << "\rConvoluted Neural Network Back Propagation ... 88%";
            auto beta_conv_0_grad = ConvBNShiftGrad(conv_act_0_err);
            UpdateScaleShift(gamma_conv_0, gamma_conv_0_grad);
            UpdateScaleShift(beta_conv_0, beta_conv_0_grad);
            cout << "\rConvoluted Neural Network Back Propagation ... 100%";
            cout << "Completed." << endl;
            cout << endl;
        }
        
    }
    return EXIT_SUCCESS;
}
