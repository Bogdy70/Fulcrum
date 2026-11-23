#pragma once

#include <fulcrum/CPUTensor.h>
#include <fulcrum/Tensor.cuh>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>

namespace fulcrum::data
{
	struct DataLoader
	{
		Tensor samples;
		Tensor labels;
		int batch_size;

		DataLoader(const Tensor& X, const Tensor& y, int batch_size = 32)
		{
			if (X.dim() != 4)
				throw runtime_error("Samples dataset must be 4 dimensional!");

			if (y.dim() != 2)
				throw runtime_error("Labels dataset must be 2 dimensional!");

			this->samples = X.clone();
			this->labels = y.clone();
			this->batch_size = batch_size;
		}

		int size() const;

		int nr_samples() const;

		int nr_labels() const;

		struct Batch
		{
			Tensor sample;
			Tensor label;
		};

		Batch operator[](int index);

		const Batch operator[](int index) const;

		struct Iterator
		{
			int position;
			DataLoader* owner;

			Batch operator*() const
			{
				return (*owner)[position];
			}

			void operator++()
			{
				++position;
			}

			bool operator!=(const Iterator& other) const
			{
				return this->position != other.position;
			}
		};

		Iterator begin() { return { 0, this }; }
		Iterator end() { return { size(), this }; }
	};

	void printMnistImage(const CPUTensor& img, int sample_idx);

	float choose_label(const Tensor& y, int idx);

	CPUTensor choose_img(const CPUTensor& X, int idx);

	CPUTensor load_img(const cv::String& path, int size = 128);

	CPUTensor loadMatrixBin(const std::string& filepath, int rows, int cols);

	CPUTensor loadTensorBin(const std::string& filepath, const std::vector<int>& shape);
}