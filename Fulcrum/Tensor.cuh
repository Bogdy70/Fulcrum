#pragma once

#include <cuda_runtime.h>
#include <vector>
#include "Common.h"
#include <variant>

using namespace std;

class CPUTensor;

struct MaxPoolRes;

struct MaxStats
{
	float value;
	int index;
};

constexpr int MAX_TENSOR_LENGTH = 8;

struct BroadcastStats
{
	int out_shape[MAX_TENSOR_LENGTH]{};
	int out_stride[MAX_TENSOR_LENGTH]{};
	int strideA[MAX_TENSOR_LENGTH]{};
	int strideB[MAX_TENSOR_LENGTH]{};
};


class Tensor
{
private:
	void* data = nullptr;
	vector<int> shape;
	vector<int> stride;
	int total = 0;
	DataType dtype = DataType::float32;

	static int calculateTotal(const vector<int>& shape);
	static vector<int> calculateStride(const vector<int>& shape);
	static size_t calcDataSize(DataType dtype);

public:
	size_t byteSize() const;

	Tensor();

	Tensor(const vector<int>& shape, DataType dtype = DataType::float32);

	~Tensor();

	Tensor(const Tensor&) = delete;
	Tensor& operator=(const Tensor&) = delete;

	Tensor(Tensor&& other) noexcept;
	Tensor& operator=(Tensor&& other) noexcept;

	float* getFloatData();

	const float* getFloatData() const;

	int32_t* getIntData();

	const int32_t* getIntData() const;

	Tensor& operator=(const std::vector<float>& X);

	Tensor& operator=(const std::vector<int32_t>& X);

	int size() const;

	int dim() const;

	const vector<int>& getShape() const;

	const vector<int>& getStride() const;

	DataType getDataType() const;

	CPUTensor toCPU() const;

	static Tensor zeros(const vector<int>& shape, DataType dtype = DataType::float32);

	static Tensor fill(const vector<int>& shape, float value, DataType dtype = DataType::float32);

	static Tensor random(const vector<int>& shape);

	static Tensor randomUniform(const vector<int>& shape, float start, float end);

	Tensor& reshape(const vector<int>& shape);

	Tensor& resize(const vector<int>& shape);

	Tensor& squeeze(int dim = -1);

	Tensor& unsqueeze(int dim = 0);

	Tensor& flatten(int start_dim = 0, int end_dim = -1);

	static Tensor pad(const Tensor& A, int padding, float val = 0.0f);

	static Tensor conv2D(const Tensor& A, const Tensor& K, int kernel_size = 3, int hStride = 1, int vStride = 1, int padding = 0);

	static Tensor conv2D_dK(const Tensor& A, const Tensor& dZ, int frd_kdim = 3, int hStride = 1, int vStride = 1, int frd_padding = 0);

	static Tensor conv2D_dX(const Tensor& dZ, const Tensor& K, int frdN, int frdM, int hStride = 1, int vStride = 1, int frd_padding = 0);

	static MaxPoolRes maxPool2D(const Tensor& A, int kernel_size = 2, int hStride = 1, int vStride = 1, int padding = 0);

	static Tensor back_maxPool2D(const Tensor& dP_vals, const Tensor& dP_idxs, int frdN, int frdM, int frd_padding = 0);

	Tensor operator*(const Tensor& B) const;

	Tensor operator/(const Tensor& B) const;

	Tensor operator+(const Tensor& B) const;

	Tensor operator-(const Tensor& B) const;

	Tensor operator*(float x) const;

	Tensor operator/(float x) const;

	Tensor operator+(float x) const;

	Tensor operator-(float x) const;

	Tensor operator==(const Tensor& B) const;

	Tensor operator>(float x) const;

	Tensor operator<(float x) const;

	Tensor matmul(const Tensor& B) const;

	Tensor T() const;

	static Tensor sum(const Tensor& A, int axis = -1, bool keepdim = false);

	static Tensor maxT(const Tensor& A, int axis = -1, bool keepdim = false);

	static Tensor argmax(const Tensor& A, int axis = -1, bool keepdim = false);

	static Tensor powT(const Tensor& A, float power);

	static Tensor sqrtT(const Tensor& A);

	static Tensor expT(const Tensor& A);

	static Tensor logT(const Tensor& A);

	static Tensor absT(const Tensor& A);

	static Tensor clipT(const Tensor& A, float minVal, float maxVal);

	static Tensor tanhT(const Tensor& A);

	static Tensor relu(const Tensor& A);

	static Tensor der_relu(const Tensor& A);

	Tensor clone() const;

	float toScalar() const;
};

Tensor operator*(float x, const Tensor& A);

Tensor operator+(float x, const Tensor& A);

Tensor operator/(float x, const Tensor& A);

Tensor operator-(float x, const Tensor& A);

struct MaxPoolRes
{
	Tensor vals;
	Tensor idxs;

	MaxPoolRes(vector<int>& shape) : vals(shape), idxs(shape, DataType::int32) {}
};