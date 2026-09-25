#pragma once

#include <vector>
#include "Common.h"
#include <string>

using namespace std;

struct CPUStorage
{
	vector<float> f;
	vector<int32_t> i;
};

class CPUTensor
{
private:
	CPUStorage data;
	vector<int> shape;
	vector<int> stride;
	int total = 0;
	DataType dtype = DataType::float32;

	static int calculateTotal(const vector<int>& shape);
	static vector<int> calculateStride(const vector<int>& shape);
	template<typename T>
	void recursivePrint(const vector<T>& vals, int dim, int offset, int indent) const;

public:
	CPUTensor();

	CPUTensor(const vector<int>& shape, DataType dtype = DataType::float32);

	float& operator()(int index);

	const float& operator()(int index) const;

	float* getFloatData();

	const float* getFloatData() const;

	int* getIntData();

	const int* getIntData() const;

	CPUTensor& operator=(const vector<float>& X);

	int size() const;

	int dim() const;

	const vector<int>& getShape() const;

	const vector<int>& getStride() const;

	DataType getDataType() const;

	Tensor toCUDA() const;

	void print() const;

	void print_dims() const;

	static void setSeed(int seed);

	static CPUTensor random(const vector<int>& shape);

	static CPUTensor randomUniform(const vector<int>& shape, float start, float end);

	static void recursMapping(vector<int>& I, const vector<int>& shape, const vector<int>& stride, int idx, int dim);

	//static CPUTensor theMax(const CPUTensor& A, int axis);

	static CPUTensor loadMatrixBin(const std::string& filepath, int rows, int cols);

	static CPUTensor loadTensorBin(const std::string& filepath, const std::vector<int>& shape);
};