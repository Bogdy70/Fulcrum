#pragma once

#include <fstream>
#include <filesystem>
#include <fulcrum/Tensor.cuh>
#include <fulcrum/CPUTensor.h>
#include <fulcrum/nn/NN.h>

using namespace fulcrum::nn;

namespace fulcrum::io
{
	void save_vector_bin(const vector<float>& x, const filesystem::path& path);

	void load_vector_bin(vector<float>& x, const filesystem::path& path);

	void save_tensor(const Tensor& T, const filesystem::path& path);

	Tensor load_tensor(const filesystem::path& path);

	void save_model(const Parameters& params, const filesystem::path& path);

	Parameters load_model(const filesystem::path& path);

	void save_cnn(const CNN_model& model, const filesystem::path& path, bool save_adam = false);

	CNN_model load_cnn(const filesystem::path& path, bool restore_adam = false);
}