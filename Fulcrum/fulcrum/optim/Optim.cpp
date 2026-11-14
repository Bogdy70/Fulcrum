#include <fulcrum/optim/Optim.h>
#include <fulcrum/nn/NN.h>

namespace fulcrum::optim
{
    AdamState::AdamState(const nn::Parameters& params) : VdW(size(params.W)), VdB(size(params.B)), SdW(size(params.W)), SdB(size(params.B))
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

    nn::Parameters& optimizer(nn::Parameters& params, const nn::Backward& grads, const float lr)
    {
        int L = size(params.W);
        for (int l = 1; l < L; l++)
        {
            params.W[l] = params.W[l] - lr * grads.dW[l];
            params.B[l] = params.B[l] - lr * grads.dB[l];
        }

        return params;
    }

    nn::Parameters& adam(nn::Parameters& params, const nn::Backward& grads, AdamState& state, const float lr, const float beta1, const float beta2, const float epsilon)
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
}