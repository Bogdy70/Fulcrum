#pragma once

#include <fulcrum/CPUTensor.h>
#include <fulcrum/Tensor.cuh>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>

namespace fulcrum::data
{
	void printMnistImage(const CPUTensor& img, int sample_idx);

	float choose_label(const Tensor& y, int idx);

	CPUTensor choose_img(const CPUTensor& X, int idx);

	CPUTensor load_img(const cv::String& path);

	CPUTensor loadMatrixBin(const std::string& filepath, int rows, int cols);

	CPUTensor loadTensorBin(const std::string& filepath, const std::vector<int>& shape);
}