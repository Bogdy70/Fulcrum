#pragma once

#include <fulcrum/Tensor.cuh>
#include <fulcrum/optim/Optim.h>
#include <fulcrum/CPUTensor.h>
#include <string>
#include <stdexcept>
#include <chrono>

using namespace fulcrum::optim;

namespace fulcrum::nn
{
    struct convStats
    {
        int in_channels;
        int out_channels;
        int kernel_size;
        int hStride;
        int vStride;
        int padding;
    };

    struct MaxPoolStats
    {
        int kernel_size;
        int hStride;
        int vStride;
        int padding;
    };

    struct Parameters
    {
        vector<Tensor> W;
        vector<Tensor> B;

        Parameters(int dim) : W(dim), B(dim) {}
    };

    struct Forward
    {
        vector<Tensor> Z;
        vector<Tensor> A;
        vector<Tensor> D;
        vector<Tensor> MPoolIdxs;

        Forward(int dim) : Z(dim), A(dim), D(dim), MPoolIdxs(dim) {}
    };

    struct Backward
    {
        vector<Tensor> dZ;
        vector<Tensor> dW;
        vector<Tensor> dB;

        Backward(int dim) : dZ(dim), dW(dim), dB(dim) {}
    };

    struct Activation
    {
        Tensor(*forward)(const Tensor&);
        Tensor(*derivate)(const Tensor&);

        Activation() : forward(nullptr), derivate(nullptr) {}

        Activation(const string& name)
        {
            if (name == "relu")
            {
                forward = Tensor::relu;
                derivate = Tensor::der_relu;
            }
            else if (name == "tanh")
            {
                forward = Tensor::tanhT;
                derivate = Tensor::der_tanh;
            }
            else
                throw runtime_error("Invalid activation type. Please choose 'relu' or 'tanh'");
        }
    };

    struct CNN_model
    {
        Parameters params;
        vector<convStats> cv_stats;
        vector<MaxPoolStats> mx_stats;
        string activ;
        AdamState state;

        CNN_model(Parameters& prm, const vector<convStats>& cv, const vector<MaxPoolStats>& mx, const string& act) : params(move(prm)), cv_stats(cv), mx_stats(mx), activ(act), state(params) {}

        CNN_model(Parameters& prm, const vector<convStats>& cv, const vector<MaxPoolStats>& mx, const string& act, AdamState& stt) : params(move(prm)), cv_stats(cv), mx_stats(mx), activ(act), state(move(stt)) {}
    };

    float cost(const Tensor& Y, const Tensor& pred, const Parameters& params, const float lambda_l1 = 0.0f, const float lambda_l2 = 0.0f);

    float accuracy(const Tensor& Y, const Tensor& pred);

    Parameters init_params(const vector<int>& dim_list);

    Parameters init_conv(const vector<convStats>& conv_stats, const vector<MaxPoolStats>& pool_stats, const Tensor& X, int output_size, const string& activation);

    Forward forward_pass(const Parameters& params, const Tensor& X, const string& activation, const float dropout = 0.0f);

    Forward conv_frdpass(const Parameters& params, const Tensor& X, const vector<convStats>& cv_stats, const vector<MaxPoolStats>& mx_stats, const string& activation, float dropout = 0.0f);

    Backward backpropagation(const Forward& frd_cache, const Parameters& params, const Tensor& Y, const string& activation, const float dropout = 0.0f, const float lambda_l1 = 0.0f, const float lambda_l2 = 0.0f);

    Backward conv_bckprop(const Forward& frd_cache, const Parameters& params, const Tensor& Y, const vector<convStats>& cv_stats, const vector<MaxPoolStats>& mx_stats, const string activation, float dropout = 0.0f, float lambda_l1 = 0.0f, float lambda_l2 = 0.0f);

    Parameters train(const Tensor& X_train,
        const Tensor& X_test,
        const Tensor& y_train,
        const Tensor& y_test,
        const vector<int>& dim_list,
        const string& activation,
        const float lr,
        const int epochs,
        const int viewing_rate,
        const float beta1 = 0.9f,
        const float beta2 = 0.999f,
        const float eps = 1e-8f,
        const float dropout = 0.0f,
        const float lambda_l1 = 0.0f,
        const float lambda_l2 = 0.0f);

    CNN_model& train_cnn(CNN_model& model,
        const Tensor& Xtrain,
        const Tensor& Xtest,
        const Tensor& ytrain,
        const Tensor& ytest,
        float lr,
        int epochs,
        int viewing_rate,
        float dropout = 0.0f,
        float lambda_l1 = 0.0f,
        float lambda_l2 = 0.0f,
        bool restore_best = false,
        bool restore_meaningful = false,
        float min_delta = 0.01f,
        bool early_stopping = false,
        int patience = 3,
        int min_epochs = 10,
        float target_loss = 0.0f,
        float target_tolerance = 0.0f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float eps = 1e-8f);

    Parameters clone_params(const Parameters& params);

    CNN_model& change_params(CNN_model& model);

    CNN_model& change_params2(CNN_model& model);

    void predict(const CPUTensor& X_test, const CPUTensor& y_test, const Parameters& params, const string& activation, int imgIdx);

    void predict_cnn(const CNN_model& model, const Tensor& X, const Tensor& y, int idx);
}