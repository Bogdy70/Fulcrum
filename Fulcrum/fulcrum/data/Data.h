#pragma once

#include <fulcrum/CPUTensor.h>
#include <fulcrum/Tensor.cuh>

namespace fulcrum::data
{
	void printMnistImage(const CPUTensor& img, int sample_idx);

	float choose_label(const Tensor& y, int idx);

	CPUTensor choose_img(const CPUTensor& X, int idx);
}