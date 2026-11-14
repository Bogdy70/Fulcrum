#pragma once

#include <fulcrum/Tensor.cuh>
#include <stdexcept>

namespace fulcrum::nn
{
    struct Parameters;
    struct Backward;
}

namespace fulcrum::optim
{
    struct AdamState
    {
        vector<Tensor> VdW;
        vector<Tensor> VdB;
        vector<Tensor> SdW;
        vector<Tensor> SdB;
        int t = 0;

        //AdamState(vector<Tensor>& vdw, vector<Tensor>& vdb, vector<Tensor>& sdw, vector<Tensor>& sdb, const int t): VdW(move(vdw)), VdB(move(vdb)), SdW(move(sdw)), SdB(move(sdb)), t(t) {}

        AdamState(const nn::Parameters& params);
    };

    nn::Parameters& optimizer(nn::Parameters& params, const nn::Backward& grads, const float lr);

    nn::Parameters& adam(nn::Parameters& params, const nn::Backward& grads, AdamState& state, const float lr, const float beta1, const float beta2, const float epsilon);
}
