#pragma once

#include <fulcrum/CPUTensor.h>
#include <fulcrum/Tensor.cuh>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>

namespace fulcrum::data
{
	struct Dataset
	{
		CPUTensor samples;
		CPUTensor labels;
		vector<string> class_labels;

		Dataset(const CPUTensor& X, const CPUTensor& y): samples(X), labels(y) {}
	};

	struct DatasetSplits
	{
		Dataset train;
		Dataset test;
		Dataset val;

		DatasetSplits(const Dataset& train, const Dataset& test, const Dataset& val): train(train), test(test), val(val) {}
	};

	struct DataLoader
	{
		CPUTensor samples;
		CPUTensor labels;
		int batch_size;

		DataLoader(const CPUTensor& X, const CPUTensor& y, int batch_size = 32)
		{
			if (X.dim() != 4)
				throw runtime_error("Samples dataset must be 4 dimensional!");

			if (y.dim() != 2)
				throw runtime_error("Labels dataset must be 2 dimensional!");

			this->samples = X;
			this->labels = y;
			this->batch_size = batch_size;
		}

		DataLoader(const Dataset& data, int batch_size = 32)
		{
			if (data.samples.dim() != 4)
				throw runtime_error("Dataset samples must be 4 dimensional!");

			if (data.labels.dim() != 2)
				throw runtime_error("Dataset labels must be 2 dimensional!");

			this->samples = data.samples;
			this->labels = data.labels;
			this->batch_size = batch_size;
		}

		int size() const;

		int nr_samples() const;

		int nr_labels() const;

		struct Batch
		{
			CPUTensor sample;
			CPUTensor label;
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

	Dataset make_dataset(const cv::String& path, const vector<string>& classes, bool resize = true, int size = 128);

	DatasetSplits train_test_val_split(const Dataset& dataset, float test = 0.1f, float val = 0.2f, bool shuffle = true, int seed = 42);
}