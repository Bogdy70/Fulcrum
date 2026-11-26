#pragma once

#include <fstream>
#include <filesystem>
#include <fulcrum/Tensor.cuh>
#include <fulcrum/CPUTensor.h>
#include <fulcrum/nn/NN.h>
#include <iostream>

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

	void save_cnn_image(
		const CPUTensor& X,
		int imgIdx,
		const std::filesystem::path& path);

	CPUTensor load_img(const cv::String& path, bool resize = true, int size = 128);

	CPUTensor loadMatrixBin(const std::string& filepath, int rows, int cols);

	CPUTensor loadTensorBin(const std::string& filepath, const std::vector<int>& shape);
}