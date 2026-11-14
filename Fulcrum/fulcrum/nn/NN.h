#pragma once

#include <fulcrum/Tensor.cuh>
#include <fulcrum/optim/Optim.h>
#include <string>
#include <stdexcept>

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
}