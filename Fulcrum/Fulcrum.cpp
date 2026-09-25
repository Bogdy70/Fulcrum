#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <chrono>
#include <stdexcept>
#include "Tensor.cuh"
#include "CPUTensor.h"

using namespace std;

Tensor sigmoid(const Tensor& A)
{
    return 1.0f / (1.0f + Tensor::expT(-1.0f * A));
}

Tensor softmax(const Tensor& A)
{
    Tensor exp_A = Tensor::expT(A);
    Tensor sum_exp = Tensor::sum(exp_A, 0);

    return exp_A / sum_exp;
}

Tensor der_tanh(const Tensor& A)
{
    return 1.0f - (Tensor::tanhT(A) * Tensor::tanhT(A));
}

Tensor sign(const Tensor& A)
{
    return (A > 0.0f) - (A < 0.0f);
}

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

float cost(const Tensor& Y, const Tensor& pred, const Parameters& params, const float lambda_l1=0.0f, const float lambda_l2=0.0f)
{
    float m = static_cast<float>(Y.getShape()[Y.dim()-1]);
    int L = size(params.W);
    float epsilon = 1e-7f;
    float cost = 0.0f;
    float sumw = 0.0f;
    Tensor clippedPred = Tensor::clipT(pred, epsilon, 1.0f - epsilon);

    if (lambda_l1 < 0.0f || lambda_l2 < 0.0f)
        throw runtime_error("Lambda value cannot be less than zero.");

    if (Y.getShape()[Y.dim()-2] > 1)
        cost = (-1.0f / m) * Tensor::sum(Y * Tensor::logT(clippedPred)).toScalar();
    else
        cost = (-1.0f / m) * Tensor::sum(Y * Tensor::logT(clippedPred) + (1 - Y) * Tensor::logT(1 - clippedPred)).toScalar();

    if (lambda_l1 > 0.0f)
    {
        sumw = 0.0f;
        for (int l = 1; l < L; l++)
        {
            sumw += Tensor::sum(Tensor::absT(params.W[l])).toScalar();
        }
        cost += (lambda_l1 / m) * sumw;
    }
    if (lambda_l2 > 0.0f)
    {
        sumw = 0.0f;
        for (int l = 1; l < L; l++)
        {
            sumw += Tensor::sum(params.W[l] * params.W[l]).toScalar();
        }
        cost += (lambda_l2 / (2.0f * m)) * sumw;
    }
    return cost;
}

float accuracy(const Tensor& Y, const Tensor& pred)
{
    if (Y.getShape()[Y.dim()-2] > 1)
    {
        Tensor true_labels = Tensor::argmax(Y, 0);
        Tensor pred_labels = Tensor::argmax(pred, 0);

        return Tensor::sum(true_labels == pred_labels).toScalar() / static_cast<float>(Y.getShape()[Y.dim() - 1]);
    }
    else
    {
        Tensor pred_labels = pred > 0.5f;
        return Tensor::sum(Y == pred_labels).toScalar() / static_cast<float>(Y.getShape()[Y.dim() - 1]);
    }
}

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

    Backward(int dim): dZ(dim), dW(dim), dB(dim) {}
};

struct AdamState
{
    vector<Tensor> VdW;
    vector<Tensor> VdB;
    vector<Tensor> SdW;
    vector<Tensor> SdB;
    int t = 0;

    AdamState(const Parameters& params) : VdW(size(params.W)), VdB(size(params.B)), SdW(size(params.W)), SdB(size(params.B))
    {
        int L = size(params.W);
        for (int l = 1; l < L; l++)
        {
            VdW[l] = Tensor::zeros(params.W[l].getShape());
            VdB[l] = Tensor::zeros(params.B[l].getShape());
            SdW[l] = Tensor::zeros(params.W[l].getShape());
            SdB[l] = Tensor::zeros(params.B[l].getShape());
        }
    }
};

struct Activation
{
    Tensor(*forward)(const Tensor&);
    Tensor(*derivate)(const Tensor&);

    Activation(): forward(nullptr), derivate(nullptr) {}

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
            derivate = der_tanh;
        }
        else
            throw runtime_error("Invalid activation type. Please choose 'relu' or 'tanh'");
    }
};

struct NetworkConfig
{
    vector<int> dims;
    Activation activation;

    NetworkConfig(const vector<int>& dim_list): dims(dim_list), activation() {}

    NetworkConfig(const vector<int>& dim_list, const string& activ_name) : dims(dim_list), activation(activ_name) {}
};

void printMnistImage(const CPUTensor& img, int sample_idx)
{
    for (int i = 0; i < 784; i++)
    {
        if (i % 28 == 0)
            cout << "\n";

        float value = img(i * img.getShape()[img.dim() - 1] + sample_idx);

        if (value > 0.7f)
            cout << "#";
        else if (value > 0.3f)
            cout << ".";
        else
            cout << " ";
    }
}

Parameters init_params(const vector<int>& dim_list)
{
    int L = size(dim_list);
    Parameters params(L);

    for (int l = 1; l < L; l++)
    {
        params.W[l] = Tensor::random({ dim_list[l], dim_list[l - 1] }) * sqrt(2.0f / static_cast<float>(dim_list[l - 1]));
        params.B[l] = Tensor::zeros({ dim_list[l], 1 });
    }

    return params;
}

Parameters init_conv(const vector<convStats>& conv_stats, const vector<MaxPoolStats>& pool_stats, const Tensor& X, int output_size, const string& activation)
{
    if (conv_stats.empty())
        throw runtime_error("At least one convolution layer is required!");

    if (conv_stats.size() != pool_stats.size())
        throw runtime_error("Every conv layer must have a maxpool layer!");

    if (output_size < 1)
        throw runtime_error("Invalid output size!");

    int L = size(conv_stats);

    Parameters params(L + 2);

    int N = X.getShape()[X.dim() - 2];
    int M = X.getShape()[X.dim() - 1];

    for (int l = 1; l <= L; l++)
    {
        float n_in = static_cast<float>(conv_stats[l - 1].in_channels * conv_stats[l - 1].kernel_size * conv_stats[l - 1].kernel_size);
        float n_out = static_cast<float>(conv_stats[l - 1].out_channels * conv_stats[l - 1].kernel_size * conv_stats[l - 1].kernel_size);

        float init = activation == "relu" ? sqrt(2.0f / n_in) : sqrt(2.0f / (n_in + n_out));

        params.W[l] = Tensor::random({ conv_stats[l - 1].out_channels, conv_stats[l - 1].in_channels, conv_stats[l - 1].kernel_size, conv_stats[l - 1].kernel_size }) * init;
        params.B[l] = Tensor::zeros({ conv_stats[l - 1].out_channels, 1, 1 });

        if (conv_stats[l - 1].kernel_size > N + 2 * conv_stats[l - 1].padding || conv_stats[l - 1].kernel_size > M + 2 * conv_stats[l - 1].padding)
            throw runtime_error("Invalid convolution kernel size!");

        if (pool_stats[l - 1].kernel_size > N + 2 * pool_stats[l - 1].padding || pool_stats[l - 1].kernel_size > M + 2 * pool_stats[l - 1].padding)
            throw runtime_error("Invalid max pool kernel size!");

        if (conv_stats[l - 1].hStride < 1)
            throw runtime_error("Invalid convolution horizontal stride value!");

        if (pool_stats[l - 1].hStride < 1)
            throw runtime_error("Invalid max pool horizontal stride value!");

        if (conv_stats[l - 1].vStride < 1)
            throw runtime_error("Invalid convolution vertical stride value!");

        if (pool_stats[l - 1].vStride < 1)
            throw runtime_error("Invalid max pool vertical stride value!");

        N = (N + 2 * conv_stats[l - 1].padding - conv_stats[l - 1].kernel_size) / conv_stats[l - 1].vStride + 1;
        N = (N + 2 * pool_stats[l - 1].padding - pool_stats[l - 1].kernel_size) / pool_stats[l - 1].vStride + 1;
        M = (M + 2 * conv_stats[l - 1].padding - conv_stats[l - 1].kernel_size) / conv_stats[l - 1].hStride + 1;
        M = (M + 2 * pool_stats[l - 1].padding - pool_stats[l - 1].kernel_size) / pool_stats[l - 1].hStride + 1;
    }

    if (N <= 0 || M <= 0)
        throw runtime_error("Invalid final shape!");

    int flat_shape = params.W[L].getShape()[0] * N * M;
    params.W[L + 1] = Tensor::random({ output_size, flat_shape }) * sqrt(2.0f / static_cast<float>(flat_shape + output_size));
    params.B[L + 1] = Tensor::zeros({ output_size, 1 });

    return params;
}

Forward forward_pass(const Parameters& params, const Tensor& X, const string& activation, const float dropout=0.0f)
{
    if (dropout < 0.0f || dropout >= 1.0f)
        throw runtime_error("Dropout value must be between [0, 1).");

    int L = size(params.W);
    Forward forward_cache(L);
    Activation activ(activation);
    Tensor(*final_activ)(const Tensor&);

    forward_cache.A[0] = X.clone();

    for (int l = 1; l < L-1; l++)
    {
        forward_cache.Z[l] = params.W[l].matmul(forward_cache.A[l - 1]) + params.B[l];
        forward_cache.A[l] = activ.forward(forward_cache.Z[l]);
        if (dropout > 0.0f)
        {
            forward_cache.D[l] = Tensor::randomUniform({ forward_cache.A[l].getShape()[forward_cache.A[l].dim() - 2], forward_cache.A[l].getShape()[forward_cache.A[l].dim() - 1] }, 0.0f, 1.0f) < (1.0f - dropout);
            forward_cache.A[l] = forward_cache.A[l] * forward_cache.D[l] / (1.0f - dropout);
        }
    }
    forward_cache.Z[L - 1] = params.W[L - 1].matmul(forward_cache.A[L - 2]) + params.B[L - 1];
    final_activ = forward_cache.Z[L - 1].getShape()[forward_cache.Z[L-1].dim()-2] > 1 ? softmax : sigmoid;
    forward_cache.A[L - 1] = final_activ(forward_cache.Z[L - 1]);

    return forward_cache;
}

Forward conv_frdpass(const Parameters& params, const Tensor& X, const vector<convStats>& cv_stats, const vector<MaxPoolStats>& mx_stats, const string& activation, float dropout = 0.0f)
{
    if (dropout < 0.0f || dropout>=1.0f)
        throw runtime_error("Invalid dropout value!");

    int L = size(params.W);

    Forward frd_cache(L);
    Activation activ(activation);
    Tensor(*final_activ)(const Tensor&);

    frd_cache.A[0] = X.clone();

    for (int l = 1; l < L-1; l++)
    {
        frd_cache.Z[l] = Tensor::conv2D(frd_cache.A[l - 1], params.W[l], cv_stats[l - 1].kernel_size, cv_stats[l - 1].hStride, cv_stats[l - 1].vStride, cv_stats[l - 1].padding) + params.B[l];
        frd_cache.A[l] = activ.forward(frd_cache.Z[l]);
        MaxPoolRes max_pool = Tensor::maxPool2D(frd_cache.A[l], mx_stats[l - 1].kernel_size, mx_stats[l - 1].hStride, mx_stats[l - 1].vStride, mx_stats[l - 1].padding);
        frd_cache.A[l] = move(max_pool.vals);
        frd_cache.MPoolIdxs[l] = move(max_pool.idxs);

        if (dropout > 0.0f)
        {
            frd_cache.D[l] = Tensor::randomUniform(frd_cache.A[l].getShape(), 0.0f, 1.0f) < (1.0f - dropout);
            frd_cache.A[l] = frd_cache.A[l] * frd_cache.D[l] / (1.0f - dropout);
        }
    }

    Tensor flat_A = frd_cache.A[L - 2].clone().flatten(1).T();
    frd_cache.Z[L - 1] = params.W[L - 1].matmul(flat_A) + params.B[L - 1];
    final_activ = frd_cache.Z[L - 1].getShape()[0] > 1 ? softmax : sigmoid;
    frd_cache.A[L - 1] = final_activ(frd_cache.Z[L - 1]);

    return frd_cache;
}

Backward backpropagation(const Forward& frd_cache, const Parameters& params, const Tensor& Y, const string& activation, const float dropout=0.0f, const float lambda_l1=0.0f, const float lambda_l2=0.0f)
{
    int L = size(frd_cache.A);
    float m = static_cast<float>(Y.getShape()[Y.dim() - 1]);
    Backward grads(L);
    Activation activ(activation);

    grads.dZ[L - 1] = frd_cache.A[L - 1] - Y;
    grads.dW[L - 1] = (1.0f / m) * grads.dZ[L - 1].matmul(frd_cache.A[L - 2].T());
    grads.dB[L - 1] = (1.0f / m) * Tensor::sum(grads.dZ[L - 1], 1, true);
    if (lambda_l1 > 0.0f)
        grads.dW[L - 1] = grads.dW[L - 1] + ((lambda_l1 / m) * sign(params.W[L - 1]));
    if (lambda_l2 > 0.0f)
        grads.dW[L - 1] = grads.dW[L - 1] + ((lambda_l2 / m) * params.W[L - 1]);

    for (int l = L - 2; l > 0; l--)
    {
        grads.dZ[l] = params.W[l + 1].T().matmul(grads.dZ[l + 1]);
        if (dropout > 0.0f)
            grads.dZ[l] = grads.dZ[l] * frd_cache.D[l] / (1.0f - dropout);
        grads.dZ[l] = grads.dZ[l] * activ.derivate(frd_cache.Z[l]);
        grads.dW[l] = (1.0f / m) * grads.dZ[l].matmul(frd_cache.A[l - 1].T());
        if (lambda_l1 > 0.0f)
            grads.dW[l] = grads.dW[l] + ((lambda_l1 / m) * sign(params.W[l]));
        if (lambda_l2 > 0.0f)
            grads.dW[l] = grads.dW[l] + ((lambda_l2 / m) * params.W[l]);
        grads.dB[l] = (1.0f / m) * Tensor::sum(grads.dZ[l], 1, true);
    }

    return grads;
}

Backward conv_bckprop(const Forward& frd_cache, const Parameters& params, const Tensor& Y, const vector<convStats>& cv_stats, const vector<MaxPoolStats>& mx_stats, const string activation, float dropout = 0.0f, float lambda_l1 = 0.0f, float lambda_l2 = 0.0f)
{
    if (dropout < 0.0f || dropout >= 1.0f)
        throw runtime_error("Invalid dropout value!");

    if (lambda_l1 < 0.0f || lambda_l2 < 0.0f)
        throw runtime_error("Lambda values cannot be less than 0!");

    int L = size(frd_cache.A);
    float m = static_cast<float>(frd_cache.A[0].getShape()[0]);
    Backward grads(L);
    Activation activ(activation);

    grads.dZ[L - 1] = frd_cache.A[L - 1] - Y;
    Tensor flat_A = frd_cache.A[L - 2].clone();
    grads.dW[L - 1] = (1.0f / m) * grads.dZ[L - 1].matmul(flat_A.flatten(1));
    if (lambda_l1 > 0.0f)
        grads.dW[L - 1] = grads.dW[L - 1] + ((lambda_l1 / m) * sign(params.W[L - 1]));
    if (lambda_l2 > 0.0f)
        grads.dW[L - 1] = grads.dW[L - 1] + ((lambda_l2 / m) * params.W[L - 1]);
    grads.dB[L - 1] = (1.0f / m) * Tensor::sum(grads.dZ[L - 1], 1, true);
    grads.dZ[L - 2] = params.W[L - 1].T().matmul(grads.dZ[L - 1]); //df
    grads.dZ[L - 2] = grads.dZ[L - 2].T();
    grads.dZ[L - 2].reshape(frd_cache.A[L - 2].getShape()); //dP

    for (int l = L - 2; l >= 1; l--)
    {
        if (dropout > 0.0f)
            grads.dZ[l] = grads.dZ[l] * frd_cache.D[l] / (1.0f - dropout);
        grads.dZ[l] = Tensor::back_maxPool2D(grads.dZ[l], frd_cache.MPoolIdxs[l], frd_cache.Z[l].getShape()[frd_cache.Z[l].dim() - 2], frd_cache.Z[l].getShape()[frd_cache.Z[l].dim() - 1], mx_stats[l - 1].padding); //dA
        grads.dZ[l] = grads.dZ[l] * activ.derivate(frd_cache.Z[l]); //dZ
        grads.dW[l] = Tensor::conv2D_dK(frd_cache.A[l - 1], grads.dZ[l], cv_stats[l - 1].kernel_size, cv_stats[l - 1].hStride, cv_stats[l - 1].vStride, cv_stats[l - 1].padding);
        if (lambda_l1 > 0.0f)
            grads.dW[l] = grads.dW[l] + ((lambda_l1 / m) * sign(params.W[l]));
        if (lambda_l2 > 0.0f)
            grads.dW[l] = grads.dW[l] + ((lambda_l2 / m) * params.W[l]);
        grads.dB[l] = (1.0f / m) * Tensor::sum(Tensor::sum(Tensor::sum(grads.dZ[l], 3, true), 2, true), 0);
        if (l != 1)
            grads.dZ[l - 1] = Tensor::conv2D_dX(grads.dZ[l], params.W[l], frd_cache.A[l - 1].getShape()[frd_cache.A[l - 1].dim() - 2], frd_cache.A[l - 1].getShape()[frd_cache.A[l - 1].dim() - 1], cv_stats[l - 1].hStride, cv_stats[l - 1].vStride, cv_stats[l - 1].padding);
    }

    return grads;
}

Parameters& optimizer(Parameters& params, const Backward& grads, const float lr)
{
    int L = size(params.W);
    for (int l = 1; l < L; l++)
    {
        params.W[l] = params.W[l] - lr * grads.dW[l];
        params.B[l] = params.B[l] - lr * grads.dB[l];
    }

    return params;
}

Parameters& adam(Parameters& params, const Backward& grads, AdamState& state, const float lr, const float beta1, const float beta2, const float epsilon)
{
    if (beta1 < 0.0f || beta1 >= 1.0f)
        throw runtime_error("Beta1 value must be between [0, 1).");
    if (beta2 < 0.0f || beta2 >= 1.0f)
        throw runtime_error("Beta2 value must be between [0, 1).");
    if (epsilon <= 0.0f)
        throw runtime_error("Epsilon must be bigger than 0.");
    int L = size(params.W);
    state.t++;
    float beta1_correction = 1.0f - powf(beta1, state.t);
    float beta2_correction = 1.0f - powf(beta2, state.t);
    for (int l = 1; l < L; l++)
    {
        state.VdW[l] = beta1 * state.VdW[l] + (1.0f - beta1) * grads.dW[l];
        state.VdB[l] = beta1 * state.VdB[l] + (1.0f - beta1) * grads.dB[l];
        state.SdW[l] = beta2 * state.SdW[l] + (1.0f - beta2) * (grads.dW[l] * grads.dW[l]);
        state.SdB[l] = beta2 * state.SdB[l] + (1.0f - beta2) * (grads.dB[l] * grads.dB[l]);

        Tensor VdW_corrected = state.VdW[l] / beta1_correction;
        Tensor VdB_corrected = state.VdB[l] / beta1_correction;

        Tensor SdW_corrected = state.SdW[l] / beta2_correction;
        Tensor SdB_corrected = state.SdB[l] / beta2_correction;

        params.W[l] = params.W[l] - lr * VdW_corrected / (Tensor::sqrtT(SdW_corrected) + epsilon);
        params.B[l] = params.B[l] - lr * VdB_corrected / (Tensor::sqrtT(SdB_corrected) + epsilon);
    }

    return params;
}

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
    const float dropout=0.0f,
    const float lambda_l1=0.0f,
    const float lambda_l2=0.0f)
{
    Parameters params = init_params(dim_list);
    AdamState state(params);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch <= epochs; epoch++)
    {   
        if (epoch % viewing_rate == 0)
        {
            cudaDeviceSynchronize();

            Forward train_eval_cache = forward_pass(params, X_train, activation);
            Forward frd_cache_test = forward_pass(params, X_test, activation);

            float train_objective = cost(y_train, train_eval_cache.A[size(dim_list) - 1], params, lambda_l1, lambda_l2);
            float train_cost = cost(y_train, train_eval_cache.A[size(dim_list) - 1], params);
            float train_acc = accuracy(y_train, train_eval_cache.A[size(dim_list) - 1]);

            float test_cost = cost(y_test, frd_cache_test.A[size(dim_list) - 1], params);
            float test_acc = accuracy(y_test, frd_cache_test.A[size(dim_list) - 1]);

            auto current_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = current_time - start_time;

            cout << "Epoch: " << epoch << " || Train objective: " << train_objective << " || Train loss : " << train_cost << " || Test loss : " << test_cost << " || Train accuracy : " << train_acc * 100.0f << " % || Test accuracy : " << test_acc * 100.0f << " % || Time : " << elapsed.count() << " sec\n";
        }

        if (epoch == epochs)
            break;

        Forward frd_cache = forward_pass(params, X_train, activation, dropout);
        Backward grads = backpropagation(frd_cache, params, y_train, activation, dropout, lambda_l1, lambda_l2);
        //optimizer(params, grads, lr);
        adam(params, grads, state, lr, beta1, beta2, eps);
    }

    return params;
}

Parameters train_cnn(const Tensor& Xtrain,
    const Tensor& Xtest,
    const Tensor& ytrain,
    const Tensor& ytest,
    const vector<convStats>& cv_stats,
    const vector<MaxPoolStats>& mx_stats,
    const string& activation,
    float lr,
    int epochs,
    int viewing_rate,
    float dropout = 0.0f,
    float lambda_l1 = 0.0f,
    float lambda_l2 = 0.0f,
    float beta1 = 0.9f,
    float beta2 = 0.999f,
    float eps = 1e-8f)
{
    Parameters cnn_params = init_conv(cv_stats, mx_stats, Xtrain, ytrain.getShape()[0], activation);
    AdamState cnn_adam(cnn_params);

    auto start_time = chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch <= epochs; epoch++)
    {
        if (epoch % viewing_rate == 0 || epoch == epochs)
        {
            cudaDeviceSynchronize();

            Forward train_frd_cache = conv_frdpass(cnn_params, Xtrain, cv_stats, mx_stats, activation);
            Forward test_frd_cache = conv_frdpass(cnn_params, Xtest, cv_stats, mx_stats, activation);

            float train_objective = cost(ytrain, train_frd_cache.A[size(cnn_params.W) - 1], cnn_params, lambda_l1, lambda_l2);
            float train_cost = cost(ytrain, train_frd_cache.A[size(cnn_params.W) - 1], cnn_params);
            float train_accuracy = accuracy(ytrain, train_frd_cache.A[size(cnn_params.W) - 1]);

            float test_cost = cost(ytest, test_frd_cache.A[size(cnn_params.W) - 1], cnn_params);
            float test_accuracy = accuracy(ytest, test_frd_cache.A[size(cnn_params.W) - 1]);

            auto current_time = chrono::high_resolution_clock::now();
            chrono::duration<double> time = current_time - start_time;

            /*int lastConv = static_cast<int>(cnn_params.W.size()) - 2;
            int last = lastConv + 1;

            cout << "\nLast conv Z:\n";
            train_frd_cache.Z[lastConv].toCPU().print();

            cout << "\nLast pooled A:\n";
            train_frd_cache.A[lastConv].toCPU().print();

            cout << "\nPredictions:\n";
            train_frd_cache.A[last].toCPU().print();*/

            cout << "Epoch: " << epoch << " || Train objective: " << train_objective << " || Train cost: " << train_cost << " || Test cost: " << test_cost << " || Train acc: " << train_accuracy * 100.0f << " % || Test acc: " << test_accuracy * 100.0f << " % || Time: " << time.count() << " sec\n";
        }

        if (epoch == epochs)
            break;

        Forward cnn_frd_cache = conv_frdpass(cnn_params, Xtrain, cv_stats, mx_stats, activation, dropout);
        Backward cnn_grads = conv_bckprop(cnn_frd_cache, cnn_params, ytrain, cv_stats, mx_stats, activation, dropout, lambda_l1, lambda_l2);
        adam(cnn_params, cnn_grads, cnn_adam, lr, beta1, beta2, eps);
    }

    return cnn_params;
}

void numerical_gradient_test(
    Parameters& params,
    const Tensor& X,
    const Tensor& Y,
    const vector<convStats>& cv_stats,
    const vector<MaxPoolStats>& mx_stats,
    const string& activation,
    float epsilon = 1e-3f,
    double abs_tolerance = 2e-3,
    double rel_tolerance = 2e-2)
{
    if (epsilon <= 0.0f)
        throw runtime_error("Epsilon must be positive!");

    auto checkCuda = [](cudaError_t error)
        {
            if (error != cudaSuccess)
                throw runtime_error(cudaGetErrorString(error));
        };

    auto readTensor = [&](const Tensor& tensor)
        {
            size_t count = 1;

            for (int dimension : tensor.getShape())
                count *= static_cast<size_t>(dimension);

            vector<float> values(count);

            checkCuda(cudaMemcpy(
                values.data(),
                tensor.getFloatData(),
                count * sizeof(float),
                cudaMemcpyDeviceToHost
            ));

            return values;
        };

    auto writeElement = [&](Tensor& tensor, size_t index, float value)
        {
            checkCuda(cudaMemcpy(
                tensor.getFloatData() + index,
                &value,
                sizeof(float),
                cudaMemcpyHostToDevice
            ));
        };

    int L = static_cast<int>(params.W.size());

    // Dropout and regularization are disabled throughout this test.
    auto calculateLoss = [&]() -> double
        {
            Forward cache = conv_frdpass(
                params, X, cv_stats, mx_stats, activation, 0.0f
            );

            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());

            return static_cast<double>(
                cost(Y, cache.A[L - 1], params, 0.0f, 0.0f)
                );
        };

    // Analytical gradients at the original parameter values.
    Forward cache = conv_frdpass(
        params, X, cv_stats, mx_stats, activation, 0.0f
    );

    Backward grads = conv_bckprop(
        cache, params, Y,
        cv_stats, mx_stats, activation, 0.0f
    );

    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    size_t totalChecked = 0;
    size_t totalFailed = 0;

    auto oldPrecision = cout.precision();
    cout << setprecision(8);
    cout << "\nNumerical gradient test\n";

    for (int l = 1; l < L; l++)
    {
        for (bool isBias : { false, true })
        {
            Tensor& parameter =
                isBias ? params.B[l] : params.W[l];

            const Tensor& gradient =
                isBias ? grads.dB[l] : grads.dW[l];

            string name =
                string(isBias ? "B[" : "W[") + to_string(l) + "]";

            if (parameter.getShape() != gradient.getShape())
                throw runtime_error(name + ": gradient shape mismatch!");

            vector<float> original = readTensor(parameter);
            vector<float> analytical = readTensor(gradient);

            size_t failed = 0;
            double largestError = 0.0;

            cout << "\n" << name << "\n";

            for (size_t i = 0; i < original.size(); i++)
            {
                // Use the actual representable float perturbations.
                float plusValue = original[i] + epsilon;
                float minusValue = original[i] - epsilon;

                if (plusValue == minusValue)
                    throw runtime_error("Epsilon too small for parameter!");

                double plusLoss;
                double minusLoss;

                try
                {
                    writeElement(parameter, i, plusValue);
                    plusLoss = calculateLoss();

                    writeElement(parameter, i, minusValue);
                    minusLoss = calculateLoss();
                }
                catch (...)
                {
                    writeElement(parameter, i, original[i]);
                    throw;
                }

                // Restore this parameter before checking the next one.
                writeElement(parameter, i, original[i]);

                double numerical =
                    (plusLoss - minusLoss) /
                    (static_cast<double>(plusValue) - minusValue);

                double backward = analytical[i];
                double error = std::abs(numerical - backward);

                double allowedError =
                    abs_tolerance +
                    rel_tolerance *
                    std::max(std::abs(numerical), std::abs(backward));

                bool passed =
                    std::isfinite(numerical) &&
                    std::isfinite(backward) &&
                    error <= allowedError;

                if (!passed)
                    failed++;

                if (std::isfinite(error))
                    largestError = std::max(largestError, error);

                // Print a few examples, plus every failure.
                if (i < 3 || !passed)
                {
                    cout << "  Element " << i
                        << " | backward: " << backward
                        << " | numerical: " << numerical
                        << " | error: " << error
                        << " | " << (passed ? "PASS" : "FAIL")
                        << "\n";
                }
            }

            totalChecked += original.size();
            totalFailed += failed;

            cout << "  Checked: " << original.size()
                << " | Failed: " << failed
                << " | Largest finite absolute error: "
                << largestError << "\n";
        }
    }

    cout << "\nOverall: "
        << (totalFailed == 0 ? "PASS" : "FAIL")
        << " | Checked: " << totalChecked
        << " | Failed: " << totalFailed << "\n";

    cout.precision(oldPrecision);
}

void predict(const CPUTensor& X_test, const CPUTensor& y_test, const Parameters& params, const string& activation, int imgIdx)
{
    CPUTensor one_mnist({ X_test.getShape()[X_test.dim() - 2], 1 });

    for (int i = 0; i < X_test.getShape()[X_test.dim()-2]; i++)
    {
        one_mnist(i) = X_test(i * X_test.getShape()[X_test.dim() - 1] + imgIdx);
    }

    CPUTensor one_mnisty({ y_test.getShape()[y_test.dim() - 2], 1 });

    for (int i = 0; i < y_test.getShape()[y_test.dim() - 2]; i++)
    {
        one_mnisty(i) = y_test(i * y_test.getShape()[y_test.dim() - 1] + imgIdx);
    }

    int L = static_cast<int>(params.W.size());

    int pred_label = 0;

    if (y_test.getShape()[y_test.dim() - 2] > 1)
    {
        Forward frd_cache1 = forward_pass(params, one_mnist.toCUDA(), activation);

        pred_label = Tensor::argmax(frd_cache1.A[L - 1]).toScalar();

        int truth_label = Tensor::argmax(one_mnisty.toCUDA()).toScalar();

        cout << "\nTruth: " << truth_label << " || Pred: " << pred_label;

        printMnistImage(X_test, imgIdx);
    }
    else
    {
        Forward frd_cache1 = forward_pass(params, one_mnist.toCUDA(), activation);

        pred_label = (frd_cache1.A[L - 1] > 0.5f).toCPU()(0);

        cout << "\nTruth: " << y_test(imgIdx) << " || Pred: " << pred_label;
    }
}

int main()
{
    try
    {
        CPUTensor X_train_cat = CPUTensor::loadMatrixBin("data/cat/X_train.bin", 12288, 209);
        CPUTensor y_train_cat = CPUTensor::loadMatrixBin("data/cat/Y_train.bin", 1, 209);

        CPUTensor X_test_cat = CPUTensor::loadMatrixBin("data/cat/X_test.bin", 12288, 50);
        CPUTensor y_test_cat = CPUTensor::loadMatrixBin("data/cat/Y_test.bin", 1, 50);

        CPUTensor X_train_mnist = CPUTensor::loadMatrixBin("data/mnist/X_train.bin", 784, 5000);
        CPUTensor y_train_mnist = CPUTensor::loadMatrixBin("data/mnist/Y_train.bin", 10, 5000);

        CPUTensor X_test_mnist = CPUTensor::loadMatrixBin("data/mnist/X_test.bin", 784, 1000);
        CPUTensor y_test_mnist = CPUTensor::loadMatrixBin("data/mnist/Y_test.bin", 10, 1000);

        CPUTensor X_train_cnn = CPUTensor::loadTensorBin("data/cifar10_cnn/X_train.bin", { 1000, 3, 32, 32 });
        CPUTensor y_train_cnn = CPUTensor::loadTensorBin("data/cifar10_cnn/Y_train.bin", { 10, 1000 });

        CPUTensor X_test_cnn = CPUTensor::loadTensorBin("data/cifar10_cnn/X_test.bin", { 100, 3, 32, 32 });
        CPUTensor y_test_cnn = CPUTensor::loadTensorBin("data/cifar10_cnn/Y_test.bin", { 10, 100 });

        CPUTensor X_train_meteors = CPUTensor::loadTensorBin("data/meteor_cnn/X_train.bin", { 1000, 3, 128, 128 });
        CPUTensor y_train_meteors = CPUTensor::loadTensorBin("data/meteor_cnn/Y_train.bin", { 1, 1000 });

        CPUTensor X_val_meteors = CPUTensor::loadTensorBin("data/meteor_cnn/X_val.bin", { 100, 3, 128, 128 });
        CPUTensor y_val_meteors = CPUTensor::loadTensorBin("data/meteor_cnn/Y_val.bin", { 1, 100 });

        CPUTensor X_test_meteors = CPUTensor::loadTensorBin("data/meteor_cnn/X_test.bin", { 100, 3, 128, 128 });
        CPUTensor y_test_meteors = CPUTensor::loadTensorBin("data/meteor_cnn/Y_test.bin", { 1, 100 });

        cout << "Cat dataset loaded successfully\n";

        cout << "X_train_cat: (" << X_train_cat.getShape()[X_train_cat.dim() - 2] << ", " << X_train_cat.getShape()[X_train_cat.dim() - 1] << ")\n";
        cout << "y_train_cat: (" << y_train_cat.getShape()[y_train_cat.dim() - 2] << ", " << y_train_cat.getShape()[y_train_cat.dim() - 1] << ")\n";
        
        cout << "X_test_cat: (" << X_test_cat.getShape()[X_test_cat.dim() - 2] << ", " << X_test_cat.getShape()[X_test_cat.dim() - 1] << ")\n";
        cout << "y_test_cat: (" << y_test_cat.getShape()[y_test_cat.dim() - 2] << ", " << y_test_cat.getShape()[y_test_cat.dim() - 1] << ")\n";

        cout << "\nMnist dataset loaded successfully\n";

        cout << "X_train_mnist: (" << X_train_mnist.getShape()[X_train_mnist.dim() - 2] << ", " << X_train_mnist.getShape()[X_train_mnist.dim() - 1] << ")\n";
        cout << "y_train_mnist: (" << y_train_mnist.getShape()[y_train_mnist.dim() - 2] << ", " << y_train_mnist.getShape()[y_train_mnist.dim() - 1] << ")\n";

        cout << "X_test_mnist: (" << X_test_mnist.getShape()[X_test_mnist.dim() - 2] << ", " << X_test_mnist.getShape()[X_test_mnist.dim() - 1] << ")\n";
        cout << "y_test_mnist: (" << y_test_mnist.getShape()[y_test_mnist.dim() - 2] << ", " << y_test_mnist.getShape()[y_test_mnist.dim() - 1] << ")\n";

        cout << "\nCifar10 dataset loaded successfully";

        cout << "\nX_train_cnn: (";
        for (int i = 0; i < X_train_cnn.dim(); i++)
        {
            cout << X_train_cnn.getShape()[i];
            if (i != X_train_cnn.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\ny_train_cnn: (";
        for (int i = 0; i < y_train_cnn.dim(); i++)
        {
            cout << y_train_cnn.getShape()[i];
            if (i != y_train_cnn.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\nX_test_cnn: (";
        for (int i = 0; i < X_test_cnn.dim(); i++)
        {
            cout << X_test_cnn.getShape()[i];
            if (i != X_test_cnn.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\ny_test_cnn: (";
        for (int i = 0; i < y_test_cnn.dim(); i++)
        {
            cout << y_test_cnn.getShape()[i];
            if (i != y_test_cnn.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\n\nMeteors dataset loaded successfully";

        cout << "\nX_train_meteors: (";
        for (int i = 0; i < X_train_meteors.dim(); i++)
        {
            cout << X_train_meteors.getShape()[i];
            if (i != X_train_meteors.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\ny_train_meteors: (";
        for (int i = 0; i < y_train_meteors.dim(); i++)
        {
            cout << y_train_meteors.getShape()[i];
            if (i != y_train_meteors.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\nX_validation_meteors: (";
        for (int i = 0; i < X_val_meteors.dim(); i++)
        {
            cout << X_val_meteors.getShape()[i];
            if (i != X_val_meteors.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\ny_validation_meteors: (";
        for (int i = 0; i < y_val_meteors.dim(); i++)
        {
            cout << y_val_meteors.getShape()[i];
            if (i != y_val_meteors.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\nX_test_meteors: (";
        for (int i = 0; i < X_test_meteors.dim(); i++)
        {
            cout << X_test_meteors.getShape()[i];
            if (i != X_test_meteors.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\ny_test_meteors: (";
        for (int i = 0; i < y_test_meteors.dim(); i++)
        {
            cout << y_test_meteors.getShape()[i];
            if (i != y_test_meteors.dim() - 1)
                cout << ", ";
        }
        cout << ")";


        vector<int> dim_list = { X_train_cat.getShape()[X_train_cat.dim() - 2], 100, 100, 200, y_train_cat.getShape()[y_train_cat.dim() - 2] };
        auto start = std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed;

        cout << "\n\nTensor test\n\n";

        Tensor T({ 2, 3, 4, 2 });
        T = vector<float> { 1, 2, 3, 4, 5, 6, 7, 8, 9, 127, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48 };
        CPUTensor CPUT = T.toCPU();
        CPUT.print();

        cout << "\n\nZeros test\n\n";

        Tensor Z = Tensor::zeros({ 2, 3, 3, 3 });
        Z.toCPU().print();


        cout << "\n\nRandom test\n\n";

        CPUTensor R = CPUTensor::random({ 2, 3, 3 });
        R.print();

        cout << "\n";

        CPUTensor RU = CPUTensor::randomUniform({ 1, 3, 3 }, 0.0f, 2.0f);
        RU.print();

        cout << "\n";

        CPUTensor::setSeed(42);
        Tensor RC = Tensor::random({ 1, 3, 3 });
        RC.toCPU().print();

        cout << "\n";
        
        CPUTensor::setSeed(42);
        Tensor RC1 = Tensor::random({ 2, 2, 3 });
        RC1.toCPU().print();

        cout << "\n\nFill test\n\n";

        Tensor F = Tensor::fill({ 3, 3, 2 }, 3.7f);
        F.toCPU().print();

        cout << "\n\nReshape test\n\n";

        Tensor S({ 2, 3, 3 });
        S = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 };
        S.toCPU().print();

        cout << "\n";

        S.reshape({ 3, 6 });
        S.toCPU().print();

        cout << "\n\nResize test\n\n";

        Tensor Rs({ 2, 3, 3 });
        Rs = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 };
        Rs.toCPU().print();

        cout << "\n";

        Rs.resize({ 2, 3, 2, 2 });
        Rs.toCPU().print();

        cout << "\n\nMul test\n\n";

        Tensor A({ 2, 3, 3 });
        A = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 };
        Tensor B({ 2, 3, 3 });
        B = vector<float>{ 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

        Tensor C = B * A;
        C.toCPU().print();

        cout << "\n\nDiv test\n\n";

        Tensor C1 = B / B;
        C1.toCPU().print();

        cout << "\n\nAdd test\n\n";

        Tensor C2 = B + A;
        C2.toCPU().print();

        cout << "\n\nSub test\n\n";

        Tensor C3 = B - A;
        C3.toCPU().print();

        cout << "\n\nScalar mul test\n\n";

        Tensor C4 = 2.0f * A;
        C4.toCPU().print();

        cout << "\n\nScalar div test\n\n";

        Tensor C5 = 2.0f / B;
        C5.toCPU().print();

        cout << "\n\nScalar add test\n\n";

        Tensor C6 = 7.0f + B;
        C6.toCPU().print();

        cout << "\n\nScalar sub test\n\n";

        Tensor C7 = 8.0f - A;
        C7.toCPU().print();
        
        cout << "\n\nSqueeze test\n\n";

        Tensor C8({ 1, 2, 3, 1 });
        C8.squeeze();
        C8.toCPU().print();
        cout << "\n(";
        for (int i = 0; i < C8.toCPU().dim(); i++)
        {
            if (i != C8.toCPU().dim() - 1)
                cout << C8.toCPU().getShape()[i] << ", ";
            else
                cout << C8.toCPU().getShape()[i];
        }
        cout << ")";

        cout << "\n\nUnsqueeze test\n\n";

        Tensor C9({4});
        C9 = vector<float>{ 1, 2, 3, 4 };
        C9.unsqueeze(1);
        C9.toCPU().print();
        cout << "\n(";
        for (int i = 0; i < C9.toCPU().dim(); i++)
        {
            if (i != C9.toCPU().dim() - 1)
                cout << C9.toCPU().getShape()[i] << ", ";
            else
                cout << C9.toCPU().getShape()[i];
        }
        cout << ")";

        cout << "\n\nPowT test\n\n";

        Tensor C10 = Tensor::powT(A, 2.0f);
        C10.toCPU().print();

        cout << "\n\nSqrtT test\n\n";

        Tensor C11 = Tensor::sqrtT(A);
        C11.toCPU().print();

        cout << "\n\nClipT test\n\n";

        Tensor C12 = Tensor::clipT(A, 3.0f, 16.0f);
        C12.toCPU().print();

        cout << "\n\nToScalar test\n\n";

        Tensor C13({ 1, 1 });
        C13 = vector<float>{ 12 };
        C13.toCPU().print();
        cout << "\n" << C13.toScalar();

        cout << "\n\nEquals test\n\n";

        Tensor D({ 2, 3, 3 });
        D = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 11, 12, 13, 27, 15, 52, 17, 18 };
        Tensor C14 = A == D;
        C14.toCPU().print();

        cout << "\n\nGreater than test\n\n";

        Tensor C15 = A > 7.0f;
        C15.toCPU().print();

        cout << "\n\nLess than test\n\n";

        Tensor C16 = B < 7.0f;
        C16.toCPU().print();

        cout << "\n\nMatmul test\n\n";

        Tensor A1({ 2, 3 });
        A1 = vector<float>{ 1, 2, 3, 4, 5, 6 };
        cout << "\n";
        A1.toCPU().print();

        Tensor B1({ 3, 4 });
        B1 = vector<float>{ 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 };
        cout << "\n";
        B1.toCPU().print();

        cout << "\n\nTranspose test\n\n";

        Tensor C18 = A1.T();
        C18.toCPU().print();

        cout << "\n\nBroadcast addition test\n\n";

        Tensor E({ 3, 1 });
        E = vector<float>{ 1, 2, 3 };

        Tensor C19 = B1 + E;
        C19.toCPU().print();

        cout << "\n\nBroadcast division test\n\n";

        Tensor F1({ 1, 4 });
        F1 = vector<float>{ 1, 2, 3, 4 };

        Tensor C20 = B1 / F1;
        C20.toCPU().print();

        cout << "\n\nSum test\n\n";

        Tensor C21 = Tensor::sum(A1);
        float x = C21.toScalar();
        cout << x << "\n\n";

        Tensor C22 = Tensor::sum(A1, 0);
        C22.toCPU().print();
        cout << "\n";

        Tensor C23 = Tensor::sum(A1, 1);
        C23.toCPU().print();

        cout << "\n\nArgmax test\n\n";

        Tensor C24 = Tensor::argmax(B1);
        C24.toCPU().print();
        cout << "\n";

        Tensor C25 = Tensor::argmax(B1, 1);
        C25.toCPU().print();

        cout << "\n\nMax test\n\n";

        cout << Tensor::maxT(B1).toScalar() << "\n\n";

        Tensor C26 = Tensor::maxT(B1, 0);
        C26.toCPU().print();
        cout << "\n";

        Tensor C27 = Tensor::maxT(B1, 1);
        C27.toCPU().print();
        cout << "\n";

        cout << "\n\nNew max test\n\n";

        Tensor G = Tensor::maxT(T, 3, true);
        G.toCPU().print();

        cout << "\n\nNew argmax test\n\n";

        Tensor Arg = Tensor::argmax(T);
        Arg.toCPU().print();

        cout << "\n\nNew sum test\n\n";

        Tensor Sum = Tensor::sum(T, 3, true);
        Sum.toCPU().print();

        cout << "\n\nMatmul test\n\n";

        Tensor T2({ 2, 3, 2, 5 });
        T2 = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 127, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59 , 60 };

        Tensor Matmul = T.matmul(T2);
        Matmul.toCPU().print();
        cout << "\n(";
        for (int i = 0; i < Matmul.dim(); i++)
        {
            cout << Matmul.getShape()[i];
            if (i != Matmul.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\n\nT test\n\n";

        Tensor Tr = T2.T();
        Tr.toCPU().print();
        cout << "\n(";
        for (int i = 0; i < Tr.dim(); i++)
        {
            cout << Tr.getShape()[i];
            if (i != Tr.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\n\nBroadcasting test\n\n";

        Tensor A2({ 3, 4, 2 });
        Tensor B2({ 3, 1, 2 });
        Tensor A3({ 2 });
        Tensor B3({ 3, 4, 2 });
        Tensor T3({ 2, 1, 4, 1, 2 });
        Tensor T4({ 1, 3, 1, 5, 2 });
        Tensor T5({ 1 });
        Tensor T6({ 1, 3, 1, 1 });
        Tensor T7({ 2, 1, 1, 5 });
        Tensor T8({ 2, 3, 4, 5 });
        Tensor T9({ 2, 3, 4, 5, 2 });
        Tensor T10({ 2, 1, 4, 1 });
        Tensor T11({ 1, 3, 1, 5 });
        Tensor T12({ 2, 3 });
        Tensor T13({ 2, 4 });

        T7 = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        T8 = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120 };
        T9 = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240 };
        T10 = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8 };
        T11 = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
        T12 = vector<float>{ 1, 2, 3, 4, 5, 6 };
        T13 = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8 };

        T6 = vector<float>{ 1, 2, 3 };

        T5 = vector<float>{ 10 };

        T3 = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
        T4 = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30 };

        Tensor T1({ 1, 2, 3, 1, 2 });
        T1 = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };

        A2 = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };
        B2 = vector<float>{ 1, 2, 3, 4, 5, 6 };
        A3 = vector<float>{ 1, 2 };
        B3 = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };

        Tensor Br = A2 + T6;
        Br.toCPU().print();
        cout << "\n(";
        for (int i = 0; i < Br.dim(); i++)
        {
            cout << Br.getShape()[i];
            if (i != Br.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\n\nTest\n\n";

        vector<int> v({ 4, 2 });
        vector<int> newv;
        newv.assign(2, 1);
        newv.insert(newv.end(), v.begin(), v.end());

        for (int i = 0; i < newv.size(); i++)
            cout << newv[i] << " ";

        cout << "\n\nPadding test\n\n";

        Tensor T14({ 3, 2, 4 });
        T14 = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };

        Tensor T15 = Tensor::pad(T9, 1);
        T15.toCPU().print();

        cout << "\n\nFlatten test\n\n";

        T14.flatten();
        T14.toCPU().print();
        cout << "\n(";
        for (int i = 0; i < T14.dim(); i++)
        {
            cout << T14.getShape()[i];
            if (i != T14.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\n\nConv2D test\n\n";

        Tensor T16({ 2, 3, 4, 4 });
        //{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48 };
        //{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
        T16 = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96 };

        CPUTensor::setSeed(42);
        Tensor K = Tensor::random({ 3, 3, 3, 3 });

        Tensor T17 = Tensor::conv2D(T16, K, 3, 2, 1, 1);
        T16.toCPU().print();
        cout << "\n";
        K.toCPU().print();
        cout << "\n";
        T17.toCPU().print();
        cout << "\n(";
        for (int i = 0; i < T17.dim(); i++)
        {
            cout << T17.getShape()[i];
            if (i != T17.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\n\nMax pool 2D test\n\n";

        MaxPoolRes mx_pl = Tensor::maxPool2D(T16, 2, 2, 2, 1);
        T16.toCPU().print();
        cout << "\n";
        mx_pl.vals.toCPU().print();
        cout << "\n(";
        for (int i = 0; i < mx_pl.vals.dim(); i++)
        {
            cout << mx_pl.vals.getShape()[i];
            if (i != mx_pl.vals.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\n\nConv init params test\n";

        vector<convStats> cv_stats = {
            {3, 16, 3, 1, 1, 0},
            {16, 32, 3, 1, 1, 0}
        };

        vector<MaxPoolStats> pl_stats = {
            {2, 2, 2, 0},
            {2, 1, 1, 0}
        };

        Tensor TX = Tensor::random({ 5000, 3, 28, 28 });
        Tensor TY = Tensor::random({ 5000, 10 });

        Parameters conv_params = init_conv(cv_stats, pl_stats, TX, TY.getShape()[1], "relu");

        conv_params.W[1].toCPU().print_dims();

        conv_params.W[2].toCPU().print_dims();

        conv_params.B[1].toCPU().print_dims();

        conv_params.B[2].toCPU().print_dims();

        conv_params.W[3].toCPU().print_dims();

        conv_params.B[3].toCPU().print_dims();

        cout << "\n\nInt32 data type test\n\n";

        Tensor Tcatva({ 2, 3 }, DataType::int32);
        Tcatva = vector<int32_t>{ 1, 2, 3, 4, 5, 6 };
        Tcatva.toCPU().print();

        cout << "\n\nFull CNN pass\n\n";
        CPUTensor::setSeed(42);
        Tensor Img = Tensor::randomUniform({ 4, 2, 8, 10 }, 0.0f, 1.0f);
        Tensor Y_Img({ 10, 4 });
        Y_Img = vector<float>{
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0
        };

        vector<convStats> cv_stats1 = {
            {2, 3, 3, 2, 1, 1},
            {3, 2, 2, 1, 1, 0}
        };

        vector<MaxPoolStats> mx_stats1 = {
            {2, 1, 2, 0},
            {2, 1, 1, 0}
        };

        Parameters cnn_params = init_conv(cv_stats1, mx_stats1, Img, 10, "relu");
        numerical_gradient_test(cnn_params, Img, Y_Img, cv_stats1, mx_stats1, "relu");
        AdamState cnn_adam1(cnn_params);
        Forward cnn_frd_cache = conv_frdpass(cnn_params, Img, cv_stats1, mx_stats1, "relu");
        Backward cnn_grads = conv_bckprop(cnn_frd_cache, cnn_params, Y_Img, cv_stats1, mx_stats1, "relu");
        auto printShape = [](const string& name, const Tensor& tensor)
            {
                const auto shape = tensor.getShape();

                cout << name << ": (";

                for (size_t i = 0; i < shape.size(); i++)
                {
                    if (i > 0)
                        cout << ", ";

                    cout << shape[i];
                }

                cout << ")\n";
            };

        printShape("Input", Img);
        printShape("Labels", Y_Img);

        int L = static_cast<int>(cnn_params.W.size());

        for (int l = 1; l < L; l++)
        {
            cout << "\nLayer " << l
                << (l == L - 1 ? " - Fully connected\n" : " - Convolution\n");

            printShape("W", cnn_params.W[l]);
            printShape("B", cnn_params.B[l]);

            printShape("Z", cnn_frd_cache.Z[l]);
            printShape("A", cnn_frd_cache.A[l]);

            if (l < L - 1)
                printShape("MaxPool indices", cnn_frd_cache.MPoolIdxs[l]);

            printShape("dZ", cnn_grads.dZ[l]);
            printShape("dW", cnn_grads.dW[l]);
            printShape("dB", cnn_grads.dB[l]);

            cout << "Weight gradient shape: "
                << (cnn_params.W[l].getShape() == cnn_grads.dW[l].getShape()
                    ? "MATCH" : "MISMATCH")
                << "\n";

            cout << "Bias gradient shape: "
                << (cnn_params.B[l].getShape() == cnn_grads.dB[l].getShape()
                    ? "MATCH" : "MISMATCH")
                << "\n";
        }
        cnn_params = move(adam(cnn_params, cnn_grads, cnn_adam1, 0.01f, 0.9f, 0.999f, 1e-8f));

        Tensor Imgtest = Img.clone();
        Tensor Y_Imgtest = Y_Img.clone();

        Parameters cnn_params_test1 = train_cnn(Img, Imgtest, Y_Img, Y_Imgtest, cv_stats1, mx_stats1, "relu", 0.001f, 1000, 100);


        cout << "\n\nCUDA Meteors dataset test\n\n";

        vector<convStats> cv_meteors = {
            {3, 8, 3, 1, 1, 1},
            {8, 16, 3, 1, 1, 1},
            {16, 32, 3, 1, 1, 1}
        };

        vector<MaxPoolStats> mx_meteors = {
            {2, 2, 2, 0},
            {2, 2, 2, 0},
            {2, 2, 2, 0}
        };

        {
            cout << "Test1: all 0\n\n";

            start = chrono::high_resolution_clock::now();

            CPUTensor::setSeed(42);

            train_cnn(X_train_meteors.toCUDA(), X_val_meteors.toCUDA(), y_train_meteors.toCUDA(), y_val_meteors.toCUDA(), cv_meteors, mx_meteors, "relu", 0.001f, 150, 17);

            cudaDeviceSynchronize();

            end = chrono::high_resolution_clock::now();

            elapsed = end - start;

            cout << "\nTest1 Meteors training time: " << elapsed.count() << " seconds\n";
        }

        {
            cout << "\nTest2: dropout = 0.2\n\n";

            start = chrono::high_resolution_clock::now();

            CPUTensor::setSeed(42);

            train_cnn(X_train_meteors.toCUDA(), X_val_meteors.toCUDA(), y_train_meteors.toCUDA(), y_val_meteors.toCUDA(), cv_meteors, mx_meteors, "relu", 0.001f, 150, 17, 0.2f);

            cudaDeviceSynchronize();

            end = chrono::high_resolution_clock::now();

            elapsed = end - start;

            cout << "\nTest2 Meteors training time: " << elapsed.count() << " seconds\n";
        }

        {
            cout << "\nTest3: L2 = 0.1\n\n";

            start = chrono::high_resolution_clock::now();

            CPUTensor::setSeed(42);

            train_cnn(X_train_meteors.toCUDA(), X_val_meteors.toCUDA(), y_train_meteors.toCUDA(), y_val_meteors.toCUDA(), cv_meteors, mx_meteors, "relu", 0.001f, 150, 17, 0.0f, 0.0f, 0.1f);

            cudaDeviceSynchronize();

            end = chrono::high_resolution_clock::now();

            elapsed = end - start;

            cout << "\nTest3 Meteors training time: " << elapsed.count() << " seconds\n";
        }

        {
            cout << "\nTest4: dropout = 0.2 + L2 = 0.1\n\n";

            start = chrono::high_resolution_clock::now();

            CPUTensor::setSeed(42);

            train_cnn(X_train_meteors.toCUDA(), X_val_meteors.toCUDA(), y_train_meteors.toCUDA(), y_val_meteors.toCUDA(), cv_meteors, mx_meteors, "relu", 0.001f, 150, 17, 0.2f, 0.0f, 0.1f);

            cudaDeviceSynchronize();

            end = chrono::high_resolution_clock::now();

            elapsed = end - start;

            cout << "\nTest4 Meteors training time: " << elapsed.count() << " seconds\n";
        }

        {
            cout << "\nTest5: tanh all 0\n\n";

            start = chrono::high_resolution_clock::now();

            CPUTensor::setSeed(42);

            train_cnn(X_train_meteors.toCUDA(), X_val_meteors.toCUDA(), y_train_meteors.toCUDA(), y_val_meteors.toCUDA(), cv_meteors, mx_meteors, "tanh", 0.001f, 150, 17);

            cudaDeviceSynchronize();

            end = chrono::high_resolution_clock::now();

            elapsed = end - start;

            cout << "\nTest5 Meteors training time: " << elapsed.count() << " seconds\n";
        }

        {
            cout << "\nTest6: tanh dropout = 0.2\n\n";

            start = chrono::high_resolution_clock::now();

            CPUTensor::setSeed(42);

            train_cnn(X_train_meteors.toCUDA(), X_val_meteors.toCUDA(), y_train_meteors.toCUDA(), y_val_meteors.toCUDA(), cv_meteors, mx_meteors, "tanh", 0.001f, 150, 17, 0.2f);

            cudaDeviceSynchronize();

            end = chrono::high_resolution_clock::now();

            elapsed = end - start;

            cout << "\nTest6 Meteors training time: " << elapsed.count() << " seconds\n";
        }

        {
            cout << "\nTest7: tanh L2 = 0.1\n\n";

            start = chrono::high_resolution_clock::now();

            CPUTensor::setSeed(42);

            train_cnn(X_train_meteors.toCUDA(), X_val_meteors.toCUDA(), y_train_meteors.toCUDA(), y_val_meteors.toCUDA(), cv_meteors, mx_meteors, "tanh", 0.001f, 150, 17, 0.0f, 0.0f, 0.1f);

            cudaDeviceSynchronize();

            end = chrono::high_resolution_clock::now();

            elapsed = end - start;

            cout << "\nTest7 Meteors training time: " << elapsed.count() << " seconds\n";
        }

        {
            cout << "\nTest8: tanh dropout = 0.2 + L2 = 0.1\n\n";

            start = chrono::high_resolution_clock::now();

            CPUTensor::setSeed(42);

            train_cnn(X_train_meteors.toCUDA(), X_val_meteors.toCUDA(), y_train_meteors.toCUDA(), y_val_meteors.toCUDA(), cv_meteors, mx_meteors, "tanh", 0.001f, 150, 17, 0.2f, 0.0f, 0.1f);

            cudaDeviceSynchronize();

            end = chrono::high_resolution_clock::now();

            elapsed = end - start;

            cout << "\nTest8 Meteors training time: " << elapsed.count() << " seconds\n";
        }
        

        cout << "\n\nCUDA cifar10 dataset test\n\n";

        start = chrono::high_resolution_clock::now();

        CPUTensor::setSeed(42);

        vector<convStats> cv_cnn = {
            {3, 8, 3, 1, 1, 1},
            {8, 16, 3, 1, 1, 1}
        };

        vector<MaxPoolStats> mx_cnn = {
            {2, 2, 2, 0},
            {2, 2, 2, 0}
        };

        Parameters cnn_params_test11 = train_cnn(X_train_cnn.toCUDA(), X_test_cnn.toCUDA(), y_train_cnn.toCUDA(), y_test_cnn.toCUDA(), cv_cnn, mx_cnn, "relu", 0.001f, 1000, 100);

        cudaDeviceSynchronize();

        end = chrono::high_resolution_clock::now();

        elapsed = end - start;

        cout << "\nCUDA cifar10 training time: " << elapsed.count() << " seconds\n";


        cout << "\n\nCUDA cat dataset test\n\n";

        start = std::chrono::high_resolution_clock::now();

        CPUTensor::setSeed(42);

        Parameters params3 = train(X_train_cat.toCUDA(), X_test_cat.toCUDA(), y_train_cat.toCUDA(), y_test_cat.toCUDA(), dim_list, "tanh", 0.0001f, 700, 100, 0.9f, 0.999f, 1e-8f, 0.2f, 0.0f, 0.01f);

        cudaDeviceSynchronize();

        end = std::chrono::high_resolution_clock::now();

        elapsed = end - start;

        cout << "\nCUDA cat training time: " << elapsed.count() << " seconds\n";

        predict(X_test_cat, y_test_cat, params3, "tanh", 7);


        dim_list = { X_train_mnist.getShape()[X_train_cat.dim() - 2], 100, 100, 200, y_train_mnist.getShape()[y_train_mnist.dim() - 2] };


        cout << "\n\nCUDA mnist dataset test\n\n";

        start = std::chrono::high_resolution_clock::now();

        CPUTensor::setSeed(123);

        Parameters params6 = train(X_train_mnist.toCUDA(), X_test_mnist.toCUDA(), y_train_mnist.toCUDA(), y_test_mnist.toCUDA(), dim_list, "relu", 0.0001f, 1000, 100, 0.9f, 0.999f, 1e-8f, 0.0f, 0.0f, 0.01f);

        cudaDeviceSynchronize();

        end = std::chrono::high_resolution_clock::now();

        elapsed = end - start;

        cout << "\nCUDA mnist training time: " << elapsed.count() << " seconds\n";
        
        predict(X_test_mnist, y_test_mnist, params6, "relu", 101);
    }
    catch (const exception& e)
    {
        cerr << "Error: " << e.what() << "\n";
    }

    return 0;
}