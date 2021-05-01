#include <iostream>
#include "pch.h"

using namespace std;
using namespace Matrix;
using namespace batnet;
using namespace dataset;
using namespace csvio;

bool IsTrained(algo_queue<Matrix::matrix> &batOutput, algo_queue<Matrix::matrix> &batOrigin, double dAcc)
{
    if(batOutput.size() == batOrigin.size())
    {
        for(auto i=0; i<batOutput.size(); i++)
        {
            auto vecDist = batOutput[i] - batOrigin[i];
            vecDist.abs_opt();
            for(auto i=0; i<vecDist.get_line(); i++) for(auto j=0; j<vecDist.get_column(); j++)
                if(vecDist[i][j] > dAcc) return false;
        }
        return true;
    }
    else exit(-1);
}

int main(int argc, char *argv[], char *envp[])
{    
    algo_queue<uint64_t> set_lbl(10);
    for(auto i=0; i<10; i++) set_lbl[i] = 10;
    mnist_set test_set("E:/VS Code project data/neural_net_batch/Res/mnist/train-images.idx3-ubyte",
    "E:/VS Code project data/neural_net_batch/Res/mnist/train-labels.idx1-ubyte", set_lbl, true);
    // test_set.output_img("E:/VS Code project data/neural_net_batch/Res/Test");
    
    auto batch_size = test_set.size();
    algo_queue<matrix> batOrigin(batch_size);
    algo_queue<algo_queue<matrix>> input(batch_size);
    for(auto i=0; i<batch_size; i++)
    {
        input[i].init();
        input[i][0] = matrix::padding(test_set[i].img, 2, 2);
        matrix lbl(10, 1);
        lbl[test_set[i].label][0] = 1;
        batOrigin[i] = lbl;
    }
    test_set.clear();

    auto kernel_0 = InitKernel(6, 1, 5, 5);
    auto gamma_0 = InitScaleBatch(1);
    auto beta_0 = InitShiftBatch(1);
    auto kernel_1 = InitKernel(16, 6, 5, 5);
    auto gamma_1 = InitScaleBatch(6);
    auto beta_1 = InitShiftBatch(6);
    auto kernel_2 = InitKernel(120, 16, 5, 5);
    auto gamma_2 = InitScaleBatch(16);
    auto beta_2 = InitShiftBatch(16);
    auto weight_0 = InitWeight(120, 84);
    double fc_gamma = 1;
    double fc_beta = 0;
    auto weight_1 = InitWeight(84, 10);
    double beta = 0.1, gamma = 0.1;

    bool flag = true;
    auto cnt = 0;
    double learn_rate = 0.03;
    double ss_rate = 1e-6;
    do
    {
        auto conv_sig_0 = ConvForwProp(input, kernel_0, 1);
        auto conv_m_sig_0 = MergeChannel(conv_sig_0);
        auto conv_bn_0 = ConvBatchNormalizationForwProp(conv_m_sig_0, beta_0, gamma_0);
        auto conv_act_0 = Activate(conv_bn_0.batScaleShift, ReLU);
        // pool0
        auto pool_0 = PoolForwProp(conv_act_0, 2, 2);
        // conv1
        auto conv_sig_1 = ConvForwProp(pool_0, kernel_1, 1);
        auto conv_m_sig_1 = MergeChannel(conv_sig_1);
        auto conv_bn_1 = ConvBatchNormalizationForwProp(conv_m_sig_1, beta_1, gamma_1);
        auto conv_act_1 = Activate(conv_bn_1.batScaleShift, ReLU);
        // pool1
        auto pool_1 = PoolForwProp(conv_act_1, 2, 2);
        // conv2
        auto conv_sig_2 = ConvForwProp(pool_1, kernel_2, 1);
        auto conv_m_sig_2 = MergeChannel(conv_sig_2);
        auto conv_bn_2 = ConvBatchNormalizationForwProp(conv_m_sig_2, beta_2, gamma_2);
        auto conv_act_2 = Activate(conv_bn_2.batScaleShift, ReLU);
        // fc 0
        auto fc_input = CNNFCNIO(conv_act_2);
        auto fc_sig = ForwProp(fc_input, weight_0);
        auto fc_bn = BatchNormalizationForwProp(fc_sig, fc_beta, fc_gamma);
        auto fc_act = Activate(fc_bn.batScaleShift, sigmoid);
        // fc 1
        auto output = GaussConnForwProp(fc_act, weight_1);

        cout << "[Round]" << cnt ++ << endl;
        for(auto i=0; i<output.size(); i++)
        {
            cout << "[Output]\t[Origin][Note]" << endl;
            for(auto j=0; j<batOrigin[i].get_line(); j++)
            {
                cout << output[i][j][0] << '\t';
                cout << batOrigin[i][j][0] << '\t' << j << endl;
            }
        }
        flag = !IsTrained(output, batOrigin, 0.2);

        if(flag)
        {
            // fc1
            auto output_err = GaussConnPreErr(fc_act, output, batOrigin, weight_1, learn_rate);
            // fc0
            // fc0_activate
            auto fc_bn_sig = DerivativeErr(output_err, fc_bn.batScaleShift, sigmoid_derivative);
            // fc0_bn
            auto fc_bn_err = BatchNormalizationPreErr(fc_bn, fc_sig, fc_bn_sig, fc_gamma);
            auto fc_gamma_grad = BNScaleGrad(output_err, fc_bn);
            fc_gamma -= fc_gamma_grad * ss_rate;
            auto fc_beta_grad = BNShiftGrad(output_err);
            fc_beta -= fc_beta_grad * ss_rate;
            // fc_0_weight
            auto weight_0_grad = WeightGrad(fc_bn_err, fc_input);
            auto fc_0_err = PreErr(fc_bn_err, weight_0);
            weight_0 -= weight_0_grad * learn_rate;
            auto fc_err = CNNFCNIO(fc_0_err);
            // conv2
            // conv_2_activate
            auto conv_2_bn_sig = ConvDerivativeErr(fc_err, conv_bn_2.batScaleShift, ReLU_derivative);
            // conv_2_bn
            auto conv_2_bn_err = ConvBatchNormalizationPreErr(conv_bn_2, conv_m_sig_2, conv_2_bn_sig, gamma_2);
            auto conv_2_gamma_grad = ConvBNScaleGrad(conv_2_bn_sig, conv_bn_2);
            UpdateScaleShift(gamma_2, conv_2_gamma_grad, ss_rate);
            auto conv_2_beta_grad = ConvBNShiftGrad(conv_2_bn_sig);
            UpdateScaleShift(beta_2, conv_2_beta_grad, ss_rate);
            // conv_2_kernel
            auto conv_2_err = ConvPreErr(conv_2_bn_err, kernel_2, 1);
            auto conv_2_kernel_grad = KernelGrad(pool_1, conv_2_bn_err);
            UpdateKernel(kernel_2, conv_2_kernel_grad, learn_rate);
            // pool_1
            auto pool_1_err = PoolPreErr(conv_2_err, conv_act_1, 2, 2);
            // conv1
            // conv_1_activate
            auto conv_1_bn_sig = ConvDerivativeErr(pool_1_err, conv_bn_1.batScaleShift, ReLU_derivative);
            // conv_1_bn
            auto conv_1_bn_err = ConvBatchNormalizationPreErr(conv_bn_1, conv_m_sig_1, conv_1_bn_sig, gamma_1);
            auto conv_1_gamma_grad = ConvBNScaleGrad(conv_1_bn_sig, conv_bn_1);
            UpdateScaleShift(gamma_1, conv_1_gamma_grad, ss_rate);
            auto conv_1_beta_grad = ConvBNShiftGrad(conv_1_bn_sig);
            UpdateScaleShift(beta_1, conv_1_beta_grad, ss_rate);
            // conv_1_kernel
            auto conv_1_err = ConvPreErr(conv_1_bn_err, kernel_1, 1);
            auto conv_1_kernel_grad = KernelGrad(pool_0, conv_1_bn_err);
            UpdateKernel(kernel_1, conv_1_kernel_grad, learn_rate);
            // pool_0
            auto pool_0_err = PoolPreErr(conv_1_err, conv_act_0, 2, 2);
            // conv0
            // conv_0_activate
            auto conv_0_bn_sig = ConvDerivativeErr(pool_0_err, conv_bn_0.batScaleShift, ReLU_derivative);
            // conv_0_bn
            auto conv_0_bn_err = ConvBatchNormalizationPreErr(conv_bn_0, conv_m_sig_0, conv_0_bn_sig, gamma_0);
            auto conv_0_gamma_grad = ConvBNScaleGrad(conv_0_bn_sig, conv_bn_0);
            UpdateScaleShift(gamma_0, conv_0_gamma_grad, ss_rate);
            auto conv_0_beta_grad = ConvBNShiftGrad(conv_0_bn_sig);
            UpdateScaleShift(beta_0, conv_0_beta_grad, ss_rate);
            // conv_0_kernel
            auto conv_0_kernel_grad = KernelGrad(input, conv_0_bn_err);
            UpdateKernel(kernel_0, conv_0_kernel_grad, learn_rate);
        }
    } while (flag);

    return EXIT_SUCCESS;
}