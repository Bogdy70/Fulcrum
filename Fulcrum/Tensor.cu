#include "Tensor.cuh"
#include "CPUTensor.h"
#include <stdexcept>

Tensor::Tensor(): data(nullptr), shape(), stride(), total(0), dtype(DataType::float32) {}

int Tensor::calculateTotal(const vector<int>& shape)
{
	int res = 1;
	for (int i = 0; i < static_cast<int>(shape.size()); i++)
	{
		if (shape[i] <= 0)
			throw runtime_error("Tensor dimensions must be pozitive.");
		res *= shape[i];
	}

	return res;
}

vector<int> Tensor::calculateStride(const vector<int>& shape)
{
	vector<int> res(shape.size());
	int val = 1;
	for (int i = static_cast<int>(shape.size()) - 1; i >= 0; i--)
	{
		res[i] = val;
		val *= shape[i];
	}

	return res;
}

size_t Tensor::calcDataSize(DataType dtype)
{
	switch (dtype)
	{
	case DataType::float32: return sizeof(float);
	case DataType::int32: return sizeof(int32_t);
	}

	throw runtime_error("Invalid data type!");
}

size_t Tensor::byteSize() const
{
	return total * calcDataSize(dtype);
}

Tensor::Tensor(const vector<int>& shape, DataType dtype): data(nullptr), shape(shape), stride(calculateStride(shape)), total(calculateTotal(shape)), dtype(dtype)
{
	cudaMalloc(&data, byteSize());
}

Tensor::~Tensor()
{
	if (data != nullptr)
		cudaFree(data);
}

Tensor::Tensor(Tensor&& other) noexcept: data(other.data), shape(move(other.shape)), stride(move(other.stride)), total(other.total), dtype(other.dtype)
{
	other.data = nullptr;
	other.shape = {};
	other.stride = {};
	other.total = 0;
	other.dtype = DataType::float32;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept
{
	if (this != &other)
	{
		if (data != nullptr)
			cudaFree(data);

		data = other.data;
		shape = move(other.shape);
		stride = move(other.stride);
		total = other.total;
		dtype = other.dtype;

		other.data = nullptr;
		other.shape = {};
		other.stride = {};
		other.total = 0;
		other.dtype = DataType::float32;
	}

	return *this;
}

float* Tensor::getFloatData()
{
	if (dtype != DataType::float32)
		throw runtime_error("Data type must be float!");

	return static_cast<float*>(data);
}

const float* Tensor::getFloatData() const
{
	if (dtype != DataType::float32)
		throw runtime_error("Data type must be float!");

	return static_cast<const float*>(data);
}

int32_t* Tensor::getIntData()
{
	if (dtype != DataType::int32)
		throw runtime_error("Data type must be int!");

	return static_cast<int32_t*>(data);
}

const int32_t* Tensor::getIntData() const
{
	if (dtype != DataType::int32)
		throw runtime_error("Data type must be int!");

	return static_cast<const int32_t*>(data);
}

Tensor& Tensor::operator=(const vector<float>& X)
{
	if (X.size() != total)
		throw runtime_error("Sizes do not match!");

	cudaMemcpy(data, X.data(), total * sizeof(float), cudaMemcpyHostToDevice);

	return *this;
}

Tensor& Tensor::operator=(const vector<int32_t>& X)
{
	if (X.size() != total)
		throw runtime_error("Sizes do not match!");

	cudaMemcpy(data, X.data(), total * sizeof(int32_t), cudaMemcpyHostToDevice);

	return *this;
}

int Tensor::size() const
{
	return total;
}

int Tensor::dim() const
{
	return static_cast<int>(shape.size());
}

const vector<int>& Tensor::getShape() const
{
	return shape;
}

const vector<int>& Tensor::getStride() const
{
	return stride;
}

DataType Tensor::getDataType() const
{
	return dtype;
}

CPUTensor Tensor::toCPU() const
{
	CPUTensor T(shape, dtype);

	switch (dtype)
	{
	case DataType::float32:
		cudaMemcpy(T.getFloatData(), getFloatData(), byteSize(), cudaMemcpyDeviceToHost);
		break;
	case DataType::int32:
		cudaMemcpy(T.getIntData(), getIntData(), byteSize(), cudaMemcpyDeviceToHost);
		break;
	default:
		throw runtime_error("Invalid data type!");
	}
	
	return T;
}

Tensor Tensor::zeros(const vector<int>& shape, DataType dtype)
{
	Tensor T(shape, dtype);
	cudaMemset(T.data, 0, T.total * calcDataSize(dtype));
	return T;
}

Tensor Tensor::random(const vector<int>& shape)
{
	return CPUTensor::random(shape).toCUDA();
}

Tensor Tensor::randomUniform(const vector<int>& shape, float start, float end)
{
	return CPUTensor::randomUniform(shape, start, end).toCUDA();
}

__global__ void floatFillKernel(float* T, int size, float value)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		T[idx] = value;
	}
}

__global__ void intFillKernel(int32_t* T, int size, int32_t value)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < size)
	{
		T[idx] = value;
	}
}

Tensor Tensor::fill(const vector<int>& shape, float value, DataType dtype)
{
	Tensor T(shape, dtype);

	int block = 256;
	int grid = (T.size() + block - 1) / block;

	if (dtype == DataType::float32)
		floatFillKernel << <grid, block >> > (T.getFloatData(), T.size(), value);
	else if (dtype == DataType::int32)
		intFillKernel << <grid, block >> > (T.getIntData(), T.total, static_cast<int32_t>(value));
	else
		throw runtime_error("Invalid data type!");

	return T;
}

Tensor& Tensor::reshape(const vector<int>& shape)
{
	if (calculateTotal(shape) != total)
		throw runtime_error("Sizes do not match!");

	this->shape = shape;
	this->stride = calculateStride(shape);

	return *this;
}

__global__ void resizeKernel(float* X, float* A, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		X[idx] = A[idx];
	}
}

Tensor& Tensor::resize(const vector<int>& shape)
{
	if (dtype != DataType::float32)
		throw runtime_error("Data type must be float!");

	Tensor X = zeros(shape, dtype);

	int block = 256;
	
	int copySize = (total <= X.total) ? total : X.total;
	int grid = (copySize + block - 1) / block;
	resizeKernel << <grid, block >> > (X.getFloatData(), getFloatData(), copySize);

	*this = move(X);

	return *this;
}

Tensor& Tensor::squeeze(int dim)
{
	vector<int> new_shape;

	if (dim<-1 || dim>static_cast<int>(shape.size())-1)
		throw runtime_error("Dim value must be between -1 and tensor dimension - 1!");

	if (dim != -1)
	{
		if (shape[dim] != 1)
		{
			return *this;
		}
		for (int i = 0; i < shape.size(); i++)
		{
			if (i != dim)
				new_shape.push_back(shape[i]);
		}
	}
	else
	{
		for (int d : shape)
		{
			if (d != 1)
				new_shape.push_back(d);
		}
	}

	shape = new_shape;
	stride = calculateStride(new_shape);

	return *this;
}

Tensor& Tensor::unsqueeze(int dim)
{
	if (dim<-1 || dim>static_cast<int>(shape.size()))
		throw runtime_error("Dim value must be between -1 and tensor dimension!");

	vector<int> new_shape;

	if (dim == -1 || dim==shape.size())
	{
		new_shape = shape;
		new_shape.push_back(1);
		shape = new_shape;
		stride = calculateStride(new_shape);
		return *this;
	}
	for (int i = 0; i < shape.size(); i++)
	{
		if (i == dim)
			new_shape.push_back(1);
		new_shape.push_back(shape[i]);
	}
	shape = new_shape;
	stride = calculateStride(new_shape);
	return *this;
}

Tensor& Tensor::flatten(int start_dim, int end_dim)
{
	if (start_dim<0 || start_dim>=static_cast<int>(shape.size()) || end_dim<-1 || end_dim>static_cast<int>(shape.size())-1)
		throw runtime_error("Invalid dimension value!");

	if (end_dim != -1 && end_dim < start_dim)
		throw runtime_error("End dim cannot be smaller than the start dim!");

	int end = end_dim == -1 ? shape.size() - 1 : end_dim;

	int flatten = 1;

	for (int i = start_dim; i <= end; i++)
	{
		flatten *= shape[i];
	}

	vector<int> newShape;

	for (int i = 0; i <start_dim; i++)
	{
		newShape.push_back(shape[i]);
	}

	newShape.push_back(flatten);

	for (int i = end + 1; i < dim(); i++)
	{
		newShape.push_back(shape[i]);
	}

	shape = newShape;
	stride = calculateStride(newShape);

	return *this;
}

__global__ void paddingKernel(float* C, const float* A, int size, int M, int pM, int N, int padding)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < size)
	{
		int idxC = idx / (N * M) * (2 * padding * pM) + (idx / M + padding) * pM + idx % M + padding;

		C[idxC] = A[idx];
	} 
}

Tensor Tensor::pad(const Tensor& A, int padding, float val)
{
	if (A.dtype != DataType::float32)
		throw runtime_error("Data type must be float!");

	if (A.dim() < 2)
		throw runtime_error("Tensor must be at least 2 dimesnional for padding!");

	if (padding < 0)
		throw runtime_error("Padding cannot be negative!");

	vector<int> newShape = A.shape;

	newShape[A.dim() - 2] += 2 * padding;
	newShape[A.dim() - 1] += 2 * padding;

	Tensor C = fill(newShape, val);

	int block = 256;

	int grid = (A.total + block - 1) / block;

	paddingKernel << <grid, block >> > (C.getFloatData(), A.getFloatData(), A.total, A.shape[A.dim() - 1], C.shape[C.dim() - 1], A.shape[A.dim() - 2], padding);
	
	return C;
}

__global__ void conv2DKernel(float* C, const float* A, const float* K, int size, int channels, int filters, int kdim, int hS, int vS, int N, int M, int rN, int rM)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < size)
	{
		float total = 0.0f;

		//int mod = filters * rN * rM;
		int k = idx / (rN * rM) % filters;
		int batch = idx / (filters * rN * rM);
		int new_idx = idx % (rN * rM) + batch * rN * rM;

		for (int j = 0; j < channels; j++)
		{
			for (int i = 0; i < kdim * kdim; i++)
			{
				int idxA = i / kdim * (M - kdim) + i + new_idx % rM * hS + new_idx / rM * (vS * M) + new_idx / (rN * rM) * (channels * N - rN * vS) * M + j * N * M;
				int idxK = i + j * kdim * kdim + k * channels * kdim * kdim;

				total += A[idxA] * K[idxK];
			}
		}
		
		C[idx] = total;
	}
}

Tensor Tensor::conv2D(const Tensor& A, const Tensor& K, int kernel_size, int hStride, int vStride, int padding)
{
	if (A.dtype != DataType::float32 || K.dtype != DataType::float32)
		throw runtime_error("Data type must be float!");

	if (A.dim() < 3 || K.dim() < 3)
		throw runtime_error("Tensors must be at least 3 dimensional for convolution!");

	if (kernel_size < 1)
		throw runtime_error("Invalid kernel size!");

	if (padding < 0)
		throw runtime_error("Inavlid padding value!");

	if (hStride < 1)
		throw runtime_error("Invalid horizontal stride value!");

	if (vStride < 1)
		throw runtime_error("Invalid vertical stride value!");

	Tensor paddedA = pad(A, padding);

	if (kernel_size > paddedA.shape[paddedA.dim() - 2] || kernel_size > paddedA.shape[paddedA.dim() - 1])
		throw runtime_error("Kernel size too big!");

	vector<int> newShape;

	for (int i = 0; i < A.dim() - 3; i++)
	{
		newShape.push_back(A.shape[i]);
	}

	int rN = (A.shape[A.dim() - 2] + 2 * padding - kernel_size) / vStride + 1;
	int rM = (A.shape[A.dim() - 1] + 2 * padding - kernel_size) / hStride + 1;
	int channels = A.shape[A.dim() - 3];

	int filters = K.shape[0];

	newShape.push_back(filters);
	newShape.push_back(rN);
	newShape.push_back(rM);

	Tensor C(newShape);

	int block = 256;

	int grid = (C.total + block - 1) / block;

	conv2DKernel << <grid, block >> > (C.getFloatData(), paddedA.getFloatData(), K.getFloatData(), C.total, channels, filters, kernel_size, hStride, vStride, paddedA.shape[A.dim() - 2], paddedA.shape[A.dim() - 1], rN, rM);

	return C;
}

__global__ void dK2DKernel(float* dK, const float* A, const float* dZ, int size, int frd_kdim, int hS, int vS, int channels, int filters, int batch_size, int N, int M, int kdimN, int kdimM)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < size)
	{
		float total = 0.0f;
		int k = idx / (channels * frd_kdim * frd_kdim);
		int ires = idx % (channels * frd_kdim * frd_kdim);

		for (int j = 0; j < batch_size; j++)
		{
			for (int i = 0; i < kdimN * kdimM; i++)
			{
				int idxA = i / kdimM * (vS * M) + i % kdimM * hS + ires % frd_kdim + ires / frd_kdim * M + ires / (frd_kdim * frd_kdim) * (N - frd_kdim) * M + j * channels * N * M;
				int idxdZ = i + k * kdimN * kdimM + j * filters * kdimN * kdimM;

				total += A[idxA] * dZ[idxdZ];
			}
		}

		dK[idx] = total / batch_size;
	}
}

Tensor Tensor::conv2D_dK(const Tensor& A, const Tensor& dZ, int frd_kdim, int hStride, int vStride, int frd_padding)
{
	if (A.dtype != DataType::float32 || dZ.dtype != DataType::float32)
		throw runtime_error("Invalid data type!");

	if (A.dim() < 3 || dZ.dim() < 3)
		throw runtime_error("Tensors must be at least 3 dimensional!");

	if (hStride < 1)
		throw runtime_error("Invalid horizontal stride value!");

	if (vStride < 1)
		throw runtime_error("Invalid vertical stride value!");

	if (frd_padding < 0)
		throw runtime_error("Invalid padding value!");

	Tensor paddedA = pad(A, frd_padding);

	int N = paddedA.shape[paddedA.dim() - 2];
	int M = paddedA.shape[paddedA.dim() - 1];
	int kdimN = dZ.shape[dZ.dim() - 2];
	int kdimM = dZ.shape[dZ.dim() - 1];

	if (kdimN > N || kdimM > M)
		throw runtime_error("dZ size too big!");

	int channels = paddedA.shape[paddedA.dim() - 3];
	int batch_size = paddedA.total / (channels * N * M);
	int filters = dZ.total / (batch_size * kdimN * kdimM);

	Tensor dK({ filters, channels, frd_kdim, frd_kdim });

	int block = 256;

	int grid = (dK.total + block - 1) / block;

	dK2DKernel << <grid, block >> > (dK.getFloatData(), paddedA.getFloatData(), dZ.getFloatData(), dK.total, frd_kdim, hStride, vStride, channels, filters, batch_size, N, M, kdimN, kdimM);

	return dK;
}

__global__ void dX2DKernel(float* dX, const float* dZ, const float* K, int size, int channels, int filters, int kdim, int hS, int vS, int N, int M, int rN, int rM)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < size)
	{
		int k = idx / (rN * rM) % filters;
		int batch = idx / (filters * rN * rM);
		int new_idx = idx % (rN * rM) + batch * rN * rM;

		for (int j = 0; j < channels; j++)
		{
			for (int i = 0; i < kdim * kdim; i++)
			{
				int idxX = i / kdim * (M - kdim) + i + new_idx % rM * hS + new_idx / rM * (vS * M) + new_idx / (rN * rM) * (channels * N - rN * vS) * M + j * N * M;
				int idxK = i + j * kdim * kdim + k * channels * kdim * kdim;

				atomicAdd(&dX[idxX], dZ[idx] * K[idxK]);
			}
		}
	}
}

__global__ void reversePaddingKernel(float* C, const float* A, int size, int M, int pM, int N, int padding)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < size)
	{
		int idxA = idx / (N * M) * (2 * padding * pM) + (idx / M + padding) * pM + idx % M + padding;

		C[idx] = A[idxA];
	}
}

Tensor Tensor::conv2D_dX(const Tensor& dZ, const Tensor& K, int frdN, int frdM, int hStride, int vStride, int frd_padding)
{
	if (dZ.dtype != DataType::float32 || K.dtype != DataType::float32)
		throw runtime_error("Invalid data type!");

	if (dZ.dim() < 3 || K.dim() != 4)
		throw runtime_error("dZ must be 3 dimensional and K must have 4 dimensions!");

	if (frdN < 1 || frdM < 1)
		throw runtime_error("Invalid forward input dimensions!");

	if (hStride < 1)
		throw runtime_error("Invalid horizontal stride value!");

	if (vStride < 1)
		throw runtime_error("Invalid vertical stride value!");

	if (frd_padding < 0)
		throw runtime_error("Invalid padding value!");

	int kdim = K.shape[K.dim() - 1];
	int dZN = dZ.shape[dZ.dim() - 2];
	int dZM = dZ.shape[dZ.dim() - 1];
	int padN = frdN + 2 * frd_padding;
	int padM = frdM + 2 * frd_padding;

	if (kdim > padN || kdim > padM)
		throw runtime_error("Kernel size too big!");

	int channels = K.shape[K.dim() - 3];
	int filters = dZ.shape[dZ.dim() - 3];
	int batch_size = dZ.total / (filters * dZN * dZM);

	Tensor paddedX = zeros({ batch_size, channels, padN, padM });

	int block = 256;

	int dXgrid = (dZ.total + block - 1) / block;

	dX2DKernel << <dXgrid, block >> > (paddedX.getFloatData(), dZ.getFloatData(), K.getFloatData(), dZ.total, channels, filters, kdim, hStride, vStride, padN, padM, dZN, dZM);

	if (frd_padding == 0)
		return paddedX;

	Tensor dX({ batch_size, channels, frdN, frdM });

	int padding_grid = (dX.total + block - 1) / block;

	reversePaddingKernel << <padding_grid, block >> > (dX.getFloatData(), paddedX.getFloatData(), dX.total, frdM, padM, frdN, frd_padding);

	return dX;
}

__global__ void maxPoolKernel(float* C_vals, int32_t* C_idxs, const float* A, int size, int kdim, int hS, int vS, int rN, int rM, int N, int M)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < size)
	{
		int firstIdxA = idx % rM * hS + idx / rM * vS * M + idx / (rN * rM) * (N - vS * rN) * M;
		
		float max_val = A[firstIdxA];

		int idx_max = firstIdxA;

		for (int i = 1; i < kdim * kdim; i++)
		{
			int idxA = i / kdim * (M - kdim) + i + idx % rM * hS + idx / rM * vS * M + idx / (rN * rM) * (N - vS * rN) * M;

			if (A[idxA] > max_val)
			{
				max_val = A[idxA];
				idx_max = idxA;
			}	
		}
		
		C_vals[idx] = max_val;
		C_idxs[idx] = idx_max;
	}
}

MaxPoolRes Tensor::maxPool2D(const Tensor& A, int kernel_size, int hStride, int vStride, int padding)
{
	if (A.dtype != DataType::float32)
		throw runtime_error("Data type must be float!");

	if (A.dim() < 3)
		throw runtime_error("Tensor must be at least 3 dimensional for max pooling!");

	if (kernel_size < 1)
		throw runtime_error("Invalid kernel size!");

	if (hStride < 1)
		throw runtime_error("Invalid horizontal stride value!");

	if (vStride < 1)
		throw runtime_error("Invalid vertical stride value!");

	if (padding < 0)
		throw runtime_error("Invalid paadding value!");

	Tensor paddedA = pad(A, padding, -numeric_limits<float>::infinity());

	if (kernel_size > paddedA.shape[paddedA.dim() - 2] || kernel_size > paddedA.shape[paddedA.dim() - 1])
		throw runtime_error("Kernel size too big!");

	vector<int> newShape;

	for (int i = 0; i < A.dim() - 2; i++)
	{
		newShape.push_back(A.shape[i]);
	}

	int rN = (A.shape[A.dim() - 2] + 2 * padding - kernel_size) / vStride + 1;
	int rM = (A.shape[A.dim() - 1] + 2 * padding - kernel_size) / hStride + 1;

	newShape.push_back(rN);
	newShape.push_back(rM);

	MaxPoolRes max_pool(newShape);

	int block = 256;

	int grid = (max_pool.vals.total + block - 1) / block;

	maxPoolKernel << <grid, block >> > (max_pool.vals.getFloatData(), max_pool.idxs.getIntData(), paddedA.getFloatData(), max_pool.vals.total, kernel_size, hStride, vStride, rN, rM, paddedA.shape[paddedA.dim() - 2], paddedA.shape[paddedA.dim() - 1]);

	return max_pool;
}

__global__ void backMaxPool2DKernel(float* dA, const float* dP_vals, const int32_t* dP_idxs, int size)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < size)
	{
		atomicAdd(&dA[dP_idxs[idx]], dP_vals[idx]);
	}
}

Tensor Tensor::back_maxPool2D(const Tensor& dP_vals, const Tensor& dP_idxs, int frdN, int frdM, int frd_padding)
{
	if (dP_vals.dtype != DataType::float32 || dP_idxs.dtype != DataType::int32)
		throw runtime_error("Invalid data types!");

	if (dP_vals.dim() != 4 || dP_idxs.dim() != 4)
		throw runtime_error("Expected 4D pooling gradients and indices!");

	if (dP_vals.shape != dP_idxs.shape)
		throw runtime_error("Pooling gradients and indices must match!");

	if (frdN < 1 || frdM < 1)
		throw runtime_error("Invalid forward input values!");

	if (frd_padding < 0)
		throw runtime_error("Invalid padding value!");

	int batch_size = dP_vals.shape[dP_vals.dim() - 4];
	int filters = dP_vals.shape[dP_vals.dim() - 3];
	int padN = frdN + 2 * frd_padding;
	int padM = frdM + 2 * frd_padding;

	Tensor paddedA = zeros({ batch_size, filters, padN, padM });

	int block = 256;

	int grid = (dP_vals.total + block - 1) / block;

	backMaxPool2DKernel << <grid, block >> > (paddedA.getFloatData(), dP_vals.getFloatData(), dP_idxs.getIntData(), dP_vals.total);

	if (frd_padding == 0)
		return paddedA;

	Tensor dA({ batch_size, filters, frdN, frdM });

	int padding_grid = (dA.total + block - 1) / block;

	reversePaddingKernel << <padding_grid, block >> > (dA.getFloatData(), paddedA.getFloatData(), dA.total, frdM, padM, frdN, frd_padding);

	return dA;
}

__global__ void mulKernel(float* C, const float* A, const float* B, int size, int subA, int subB, int upperA, int upperB, int strideA, int strideB)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < size)
	{
		int idxA = subA == -1 ? idx : idx / strideA % upperA * subA + idx % subA;
		int idxB = subB == -1 ? idx : idx / strideB % upperB * subB + idx % subB;

		C[idx] = A[idxA] * B[idxB];
	}
}

__global__ void universalMulKernel(float* C, const float* A, const float* B, int size, int dim, BroadcastStats stats)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < size)
	{
		int idxA = 0;
		int idxB = 0;

		for (int i = 0; i < dim; i++)
		{
			int coord = idx / stats.out_stride[i] % stats.out_shape[i];

			idxA += coord * stats.strideA[i];

			idxB += coord * stats.strideB[i];
		}

		C[idx] = A[idxA] * B[idxB];
	}
}

Tensor Tensor::operator*(const Tensor& B) const
{
	if (dtype != DataType::float32)
		throw runtime_error("Data type must be float!");

	vector<int> newShape = dim() >= B.dim() ? shape : B.shape;

	if (newShape.size() > MAX_TENSOR_LENGTH)
		throw runtime_error("Tensor length exceeded!");

	int subDimsB, upperDimsB, upperStrideC_B;
	int subDimsA, upperDimsA, upperStrideC_A;

	BroadcastStats stats;

	int block = 256;

	if (dim() > B.dim())
	{
		vector<int> newB;
		newB.assign(dim() - B.dim(), 1);

		newB.insert(newB.end(), B.shape.begin(), B.shape.end());

		for (int i = 0; i < dim(); i++)
		{
			if (shape[i] != newB[i] && shape[i] != 1 && newB[i] != 1)
				throw runtime_error("Invalid Tensor shape!");

			if (shape[i] != 1)
				newShape[i] = shape[i];
			else if (newB[i] != 1)
				newShape[i] = newB[i];
			else
				newShape[i] = 1;
		}

		Tensor C(newShape);

		bool prev_stateA = 0;
		bool prev_stateB = 0;

		vector<int> regionsA;
		vector<int> regionsB;

		for (int i = 0; i < dim(); i++)
		{
			if (newShape[i] == 1)
				continue;

			bool current_stateA = shape[i] == 1;
			bool current_stateB = newB[i] == 1;

			if (current_stateA && !prev_stateA)
			{
				regionsA.push_back(i);
			}
			if (current_stateB && !prev_stateB)
			{
				regionsB.push_back(i);
			}

			prev_stateA = current_stateA;
			prev_stateB = current_stateB;
		}

		int grid = (C.total + block - 1) / block;

		bool supportA = regionsA.size() <= 1 || (regionsA.size() == 2 && regionsA[0] == 0);
		bool supportB = regionsB.size() <= 1 || (regionsB.size() == 2 && regionsB[0] == 0);

		if (supportA && supportB)
		{
			int axisA = -1;
			int axisB = -1;

			if (regionsA.size()) axisA = regionsA.size() == 1 ? regionsA[0] : regionsA[1];
			if (regionsB.size()) axisB = regionsB.size() == 1 ? regionsB[0] : regionsB[1];

			if (axisA == -1 && axisB == -1)
			{
				mulKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, -1, -1, -1, -1, -1, -1);
			}
			else if (axisA == -1)
			{
				subDimsB = calculateStride(newB)[axisB];
				upperDimsB = calculateTotal(newB) / subDimsB;
				upperStrideC_B = axisB == 0 ? 1 : C.stride[axisB - 1];

				mulKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, -1, subDimsB, -1, upperDimsB, -1, upperStrideC_B);
			}
			else if (axisB == -1)
			{
				subDimsA = stride[axisA];
				upperDimsA = total / subDimsA;
				upperStrideC_A = axisA == 0 ? 1 : C.stride[axisA - 1];

				mulKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, subDimsA, -1, upperDimsA, -1, upperStrideC_A, -1);
			}
			else
			{
				subDimsB = calculateStride(newB)[axisB];
				upperDimsB = calculateTotal(newB) / subDimsB;
				upperStrideC_B = axisB == 0 ? 1 : C.stride[axisB - 1];

				subDimsA = stride[axisA];
				upperDimsA = total / subDimsA;
				upperStrideC_A = axisA == 0 ? 1 : C.stride[axisA - 1];

				mulKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, subDimsA, subDimsB, upperDimsA, upperDimsB, upperStrideC_A, upperStrideC_B);
			}
		}
		else
		{
			vector<int> newStride;
			newStride.assign(dim() - B.dim(), 0);
			newStride.insert(newStride.end(), B.stride.begin(), B.stride.end());

			for (int i = 0; i < dim(); i++)
			{
				stats.strideB[i] = newB[i] != 1 ? newStride[i] : 0;
				stats.strideA[i] = shape[i] != 1 ? stride[i] : 0;
				stats.out_stride[i] = C.stride[i];
				stats.out_shape[i] = C.shape[i];
			}

			universalMulKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, C.dim(), stats);
		}

		return C;
	}
	else if (B.dim() > dim())
	{
		vector<int> newA;
		newA.assign(B.dim() - dim(), 1);

		newA.insert(newA.end(), shape.begin(), shape.end());

		for (int i = 0; i < B.dim(); i++)
		{
			if (newA[i] != B.shape[i] && newA[i] != 1 && B.shape[i] != 1)
				throw runtime_error("Invalid Tensor shape!");

			if (newA[i] != 1)
				newShape[i] = newA[i];
			else if (B.shape[i] != 1)
				newShape[i] = B.shape[i];
			else
				newShape[i] = 1;
		}

		Tensor C(newShape);

		int grid = (C.total + block - 1) / block;

		vector<int> regionsA;
		vector<int> regionsB;

		bool prev_stateA = 0;
		bool prev_stateB = 0;

		for (int i = 0; i < B.dim(); i++)
		{
			if (newShape[i] == 1)
				continue;

			bool current_stateA = newA[i] == 1;
			bool current_stateB = B.shape[i] == 1;

			if (current_stateA && !prev_stateA)
				regionsA.push_back(i);

			if (current_stateB && !prev_stateB)
				regionsB.push_back(i);

			prev_stateA = current_stateA;
			prev_stateB = current_stateB;
		}

		bool supportA = regionsA.size() <= 1 || (regionsA.size() == 2 && regionsA[0] == 0);
		bool supportB = regionsB.size() <= 1 || (regionsB.size() == 2 && regionsB[0] == 0);

		if (supportA && supportB)
		{
			int axisA = -1;
			int axisB = -1;

			if (regionsA.size()) axisA = regionsA.size() == 1 ? regionsA[0] : regionsA[1];
			if (regionsB.size()) axisB = regionsB.size() == 1 ? regionsB[0] : regionsB[1];

			if (axisA == -1 && axisB == -1)
			{
				mulKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, -1, -1, -1, -1, -1, -1);
			}
			else if (axisA == -1)
			{
				subDimsB = B.stride[axisB];
				upperDimsB = B.total / subDimsB;
				upperStrideC_B = axisB == 0 ? 1 : C.stride[axisB - 1];

				mulKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, -1, subDimsB, -1, upperDimsB, -1, upperStrideC_B);
			}
			else if (axisB == -1)
			{
				subDimsA = calculateStride(newA)[axisA];
				upperDimsA = calculateTotal(newA) / subDimsA;
				upperStrideC_A = axisA == 0 ? 1 : C.stride[axisA - 1];

				mulKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, subDimsA, -1, upperDimsA, -1, upperStrideC_A, -1);
			}
			else
			{
				subDimsA = calculateStride(newA)[axisA];
				upperDimsA = calculateTotal(newA) / subDimsA;
				upperStrideC_A = axisA == 0 ? 1 : C.stride[axisA - 1];

				subDimsB = B.stride[axisB];
				upperDimsB = B.total / subDimsB;
				upperStrideC_B = axisB == 0 ? 1 : C.stride[axisB - 1];

				mulKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, subDimsA, subDimsB, upperDimsA, upperDimsB, upperStrideC_A, upperStrideC_B);
			}
		}
		else
		{
			vector<int> newStride;
			newStride.assign(B.dim() - dim(), 0);
			newStride.insert(newStride.end(), stride.begin(), stride.end());

			for (int i = 0; i < B.dim(); i++)
			{
				stats.strideA[i] = newA[i] != 1 ? newStride[i] : 0;
				stats.strideB[i] = B.shape[i] != 1 ? B.stride[i] : 0;
				stats.out_stride[i] = C.stride[i];
				stats.out_shape[i] = C.shape[i];
			}

			universalMulKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, C.dim(), stats);
		}

		return C;
	}
	else
	{
		for (int i = 0; i < dim(); i++)
		{
			if (shape[i] != B.shape[i] && shape[i] != 1 && B.shape[i] != 1)
				throw runtime_error("Invalid Tensor shape!");

			if (shape[i] != 1)
				newShape[i] = shape[i];
			else if (B.shape[i] != 1)
				newShape[i] = B.shape[i];
			else
				newShape[i] = 1;
		}

		Tensor C(newShape);

		int grid = (C.total + block - 1) / block;

		bool prev_stateA = 0;
		bool prev_stateB = 0;

		vector<int> regionsA;
		vector<int> regionsB;

		for (int i = 0; i < dim(); i++)
		{
			if (newShape[i] == 1)
				continue;

			bool current_stateA = shape[i] == 1;
			bool current_stateB = B.shape[i] == 1;

			if (current_stateA && !prev_stateA)
				regionsA.push_back(i);

			if (current_stateB && !prev_stateB)
				regionsB.push_back(i);

			prev_stateA = current_stateA;
			prev_stateB = current_stateB;
		}

		bool supportA = regionsA.size() <= 1 || (regionsA.size() == 2 && regionsA[0] == 0);
		bool supportB = regionsB.size() <= 1 || (regionsB.size() == 2 && regionsB[0] == 0);

		if (supportA && supportB)
		{
			int axisA = -1;
			int axisB = -1;

			if (regionsA.size()) axisA = regionsA.size() == 1 ? regionsA[0] : regionsA[1];
			if (regionsB.size()) axisB = regionsB.size() == 1 ? regionsB[0] : regionsB[1];

			if (axisA == -1 && axisB == -1)
			{
				mulKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, -1, -1, -1, -1, -1, -1);
			}
			else if (axisA == -1)
			{
				subDimsB = B.stride[axisB];
				upperDimsB = B.total / subDimsB;
				upperStrideC_B = axisB == 0 ? 1 : C.stride[axisB - 1];

				mulKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, -1, subDimsB, -1, upperDimsB, -1, upperStrideC_B);
			}
			else if (axisB == -1)
			{
				subDimsA = stride[axisA];
				upperDimsA = total / subDimsA;
				upperStrideC_A = axisA == 0 ? 1 : C.stride[axisA - 1];

				mulKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, subDimsA, -1, upperDimsA, -1, upperStrideC_A, -1);
			}
			else
			{
				subDimsA = stride[axisA];
				upperDimsA = total / subDimsA;
				upperStrideC_A = axisA == 0 ? 1 : C.stride[axisA - 1];

				subDimsB = B.stride[axisB];
				upperDimsB = B.total / subDimsB;
				upperStrideC_B = axisB == 0 ? 1 : C.stride[axisB - 1];

				mulKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, subDimsA, subDimsB, upperDimsA, upperDimsB, upperStrideC_A, upperStrideC_B);
			}
		}
		else
		{
			for (int i = 0; i < dim(); i++)
			{
				stats.strideA[i] = shape[i] != 1 ? stride[i] : 0;
				stats.strideB[i] = B.shape[i] != 1 ? B.stride[i] : 0;
				stats.out_stride[i] = C.stride[i];
				stats.out_shape[i] = C.shape[i];
			}

			universalMulKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, C.dim(), stats);
		}

		return C;
	}
}

__global__ void divKernel(float* C, const float* A, const float* B, int size, int subA, int subB, int upperA, int upperB, int strideA, int strideB)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < size)
	{
		int idxA = subA == -1 ? idx : idx / strideA % upperA * subA + idx % subA;
		int idxB = subB == -1 ? idx : idx / strideB % upperB * subB + idx % subB;

		C[idx] = A[idxA] / B[idxB];
	}
}

__global__ void universalDivKernel(float* C, const float* A, const float* B, int size, int dim, BroadcastStats stats)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < size)
	{
		int idxA = 0;
		int idxB = 0;

		for (int i = 0; i < dim; i++)
		{
			int coord = idx / stats.out_stride[i] % stats.out_shape[i];

			idxA += coord * stats.strideA[i];

			idxB += coord * stats.strideB[i];
		}

		C[idx] = A[idxA] / B[idxB];
	}
}

Tensor Tensor::operator/(const Tensor& B) const
{
	vector<int> newShape = dim() >= B.dim() ? shape : B.shape;

	if (newShape.size() > MAX_TENSOR_LENGTH)
		throw runtime_error("Tensor length exceeded!");

	int subDimsB, upperDimsB, upperStrideC_B;
	int subDimsA, upperDimsA, upperStrideC_A;

	BroadcastStats stats;

	int block = 256;

	if (dim() > B.dim())
	{
		vector<int> newB;
		newB.assign(dim() - B.dim(), 1);

		newB.insert(newB.end(), B.shape.begin(), B.shape.end());

		for (int i = 0; i < dim(); i++)
		{
			if (shape[i] != newB[i] && shape[i] != 1 && newB[i] != 1)
				throw runtime_error("Invalid Tensor shape!");

			if (shape[i] != 1)
				newShape[i] = shape[i];
			else if (newB[i] != 1)
				newShape[i] = newB[i];
			else
				newShape[i] = 1;
		}

		Tensor C(newShape);

		bool prev_stateA = 0;
		bool prev_stateB = 0;

		vector<int> regionsA;
		vector<int> regionsB;

		for (int i = 0; i < dim(); i++)
		{
			if (newShape[i] == 1)
				continue;

			bool current_stateA = shape[i] == 1;
			bool current_stateB = newB[i] == 1;

			if (current_stateA && !prev_stateA)
			{
				regionsA.push_back(i);
			}
			if (current_stateB && !prev_stateB)
			{
				regionsB.push_back(i);
			}

			prev_stateA = current_stateA;
			prev_stateB = current_stateB;
		}

		int grid = (C.total + block - 1) / block;

		bool supportA = regionsA.size() <= 1 || (regionsA.size() == 2 && regionsA[0] == 0);
		bool supportB = regionsB.size() <= 1 || (regionsB.size() == 2 && regionsB[0] == 0);

		if (supportA && supportB)
		{
			int axisA = -1;
			int axisB = -1;

			if (regionsA.size()) axisA = regionsA.size() == 1 ? regionsA[0] : regionsA[1];
			if (regionsB.size()) axisB = regionsB.size() == 1 ? regionsB[0] : regionsB[1];

			if (axisA == -1 && axisB == -1)
			{
				divKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, -1, -1, -1, -1, -1, -1);
			}
			else if (axisA == -1)
			{
				subDimsB = calculateStride(newB)[axisB];
				upperDimsB = calculateTotal(newB) / subDimsB;
				upperStrideC_B = axisB == 0 ? 1 : C.stride[axisB - 1];

				divKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, -1, subDimsB, -1, upperDimsB, -1, upperStrideC_B);
			}
			else if (axisB == -1)
			{
				subDimsA = stride[axisA];
				upperDimsA = total / subDimsA;
				upperStrideC_A = axisA == 0 ? 1 : C.stride[axisA - 1];

				divKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, subDimsA, -1, upperDimsA, -1, upperStrideC_A, -1);
			}
			else
			{
				subDimsB = calculateStride(newB)[axisB];
				upperDimsB = calculateTotal(newB) / subDimsB;
				upperStrideC_B = axisB == 0 ? 1 : C.stride[axisB - 1];

				subDimsA = stride[axisA];
				upperDimsA = total / subDimsA;
				upperStrideC_A = axisA == 0 ? 1 : C.stride[axisA - 1];

				divKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, subDimsA, subDimsB, upperDimsA, upperDimsB, upperStrideC_A, upperStrideC_B);
			}
		}
		else
		{
			vector<int> newStride;
			newStride.assign(dim() - B.dim(), 0);
			newStride.insert(newStride.end(), B.stride.begin(), B.stride.end());

			for (int i = 0; i < dim(); i++)
			{
				stats.strideB[i] = newB[i] != 1 ? newStride[i] : 0;
				stats.strideA[i] = shape[i] != 1 ? stride[i] : 0;
				stats.out_stride[i] = C.stride[i];
				stats.out_shape[i] = C.shape[i];
			}

			universalDivKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, C.dim(), stats);
		}

		return C;
	}
	else if (B.dim() > dim())
	{
		vector<int> newA;
		newA.assign(B.dim() - dim(), 1);

		newA.insert(newA.end(), shape.begin(), shape.end());

		for (int i = 0; i < B.dim(); i++)
		{
			if (newA[i] != B.shape[i] && newA[i] != 1 && B.shape[i] != 1)
				throw runtime_error("Invalid Tensor shape!");

			if (newA[i] != 1)
				newShape[i] = newA[i];
			else if (B.shape[i] != 1)
				newShape[i] = B.shape[i];
			else
				newShape[i] = 1;
		}

		Tensor C(newShape);

		int grid = (C.total + block - 1) / block;

		vector<int> regionsA;
		vector<int> regionsB;

		bool prev_stateA = 0;
		bool prev_stateB = 0;

		for (int i = 0; i < B.dim(); i++)
		{
			if (newShape[i] == 1)
				continue;

			bool current_stateA = newA[i] == 1;
			bool current_stateB = B.shape[i] == 1;

			if (current_stateA && !prev_stateA)
				regionsA.push_back(i);

			if (current_stateB && !prev_stateB)
				regionsB.push_back(i);

			prev_stateA = current_stateA;
			prev_stateB = current_stateB;
		}

		bool supportA = regionsA.size() <= 1 || (regionsA.size() == 2 && regionsA[0] == 0);
		bool supportB = regionsB.size() <= 1 || (regionsB.size() == 2 && regionsB[0] == 0);

		if (supportA && supportB)
		{
			int axisA = -1;
			int axisB = -1;

			if (regionsA.size()) axisA = regionsA.size() == 1 ? regionsA[0] : regionsA[1];
			if (regionsB.size()) axisB = regionsB.size() == 1 ? regionsB[0] : regionsB[1];

			if (axisA == -1 && axisB == -1)
			{
				divKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, -1, -1, -1, -1, -1, -1);
			}
			else if (axisA == -1)
			{
				subDimsB = B.stride[axisB];
				upperDimsB = B.total / subDimsB;
				upperStrideC_B = axisB == 0 ? 1 : C.stride[axisB - 1];

				divKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, -1, subDimsB, -1, upperDimsB, -1, upperStrideC_B);
			}
			else if (axisB == -1)
			{
				subDimsA = calculateStride(newA)[axisA];
				upperDimsA = calculateTotal(newA) / subDimsA;
				upperStrideC_A = axisA == 0 ? 1 : C.stride[axisA - 1];

				divKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, subDimsA, -1, upperDimsA, -1, upperStrideC_A, -1);
			}
			else
			{
				subDimsA = calculateStride(newA)[axisA];
				upperDimsA = calculateTotal(newA) / subDimsA;
				upperStrideC_A = axisA == 0 ? 1 : C.stride[axisA - 1];

				subDimsB = B.stride[axisB];
				upperDimsB = B.total / subDimsB;
				upperStrideC_B = axisB == 0 ? 1 : C.stride[axisB - 1];

				divKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, subDimsA, subDimsB, upperDimsA, upperDimsB, upperStrideC_A, upperStrideC_B);
			}
		}
		else
		{
			vector<int> newStride;
			newStride.assign(B.dim() - dim(), 0);
			newStride.insert(newStride.end(), stride.begin(), stride.end());

			for (int i = 0; i < B.dim(); i++)
			{
				stats.strideA[i] = newA[i] != 1 ? newStride[i] : 0;
				stats.strideB[i] = B.shape[i] != 1 ? B.stride[i] : 0;
				stats.out_stride[i] = C.stride[i];
				stats.out_shape[i] = C.shape[i];
			}

			universalDivKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, C.dim(), stats);
		}

		return C;
	}
	else
	{
		for (int i = 0; i < dim(); i++)
		{
			if (shape[i] != B.shape[i] && shape[i] != 1 && B.shape[i] != 1)
				throw runtime_error("Invalid Tensor shape!");

			if (shape[i] != 1)
				newShape[i] = shape[i];
			else if (B.shape[i] != 1)
				newShape[i] = B.shape[i];
			else
				newShape[i] = 1;
		}

		Tensor C(newShape);

		int grid = (C.total + block - 1) / block;

		bool prev_stateA = 0;
		bool prev_stateB = 0;

		vector<int> regionsA;
		vector<int> regionsB;

		for (int i = 0; i < dim(); i++)
		{
			if (newShape[i] == 1)
				continue;

			bool current_stateA = shape[i] == 1;
			bool current_stateB = B.shape[i] == 1;

			if (current_stateA && !prev_stateA)
				regionsA.push_back(i);

			if (current_stateB && !prev_stateB)
				regionsB.push_back(i);

			prev_stateA = current_stateA;
			prev_stateB = current_stateB;
		}

		bool supportA = regionsA.size() <= 1 || (regionsA.size() == 2 && regionsA[0] == 0);
		bool supportB = regionsB.size() <= 1 || (regionsB.size() == 2 && regionsB[0] == 0);

		if (supportA && supportB)
		{
			int axisA = -1;
			int axisB = -1;

			if (regionsA.size()) axisA = regionsA.size() == 1 ? regionsA[0] : regionsA[1];
			if (regionsB.size()) axisB = regionsB.size() == 1 ? regionsB[0] : regionsB[1];

			if (axisA == -1 && axisB == -1)
			{
				divKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, -1, -1, -1, -1, -1, -1);
			}
			else if (axisA == -1)
			{
				subDimsB = B.stride[axisB];
				upperDimsB = B.total / subDimsB;
				upperStrideC_B = axisB == 0 ? 1 : C.stride[axisB - 1];

				divKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, -1, subDimsB, -1, upperDimsB, -1, upperStrideC_B);
			}
			else if (axisB == -1)
			{
				subDimsA = stride[axisA];
				upperDimsA = total / subDimsA;
				upperStrideC_A = axisA == 0 ? 1 : C.stride[axisA - 1];

				divKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, subDimsA, -1, upperDimsA, -1, upperStrideC_A, -1);
			}
			else
			{
				subDimsA = stride[axisA];
				upperDimsA = total / subDimsA;
				upperStrideC_A = axisA == 0 ? 1 : C.stride[axisA - 1];

				subDimsB = B.stride[axisB];
				upperDimsB = B.total / subDimsB;
				upperStrideC_B = axisB == 0 ? 1 : C.stride[axisB - 1];

				divKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, subDimsA, subDimsB, upperDimsA, upperDimsB, upperStrideC_A, upperStrideC_B);
			}
		}
		else
		{
			for (int i = 0; i < dim(); i++)
			{
				stats.strideA[i] = shape[i] != 1 ? stride[i] : 0;
				stats.strideB[i] = B.shape[i] != 1 ? B.stride[i] : 0;
				stats.out_stride[i] = C.stride[i];
				stats.out_shape[i] = C.shape[i];
			}

			universalDivKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, C.dim(), stats);
		}

		return C;
	}
}

__global__ void addKernel(float* C, const float* A, const float* B, int size, int subA, int subB, int upperA, int upperB, int strideA, int strideB)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < size)
	{
		int idxA = subA == -1 ? idx : idx / strideA % upperA * subA + idx % subA;
		int idxB = subB == -1 ? idx : idx / strideB % upperB * subB + idx % subB;

		C[idx] = A[idxA] + B[idxB];
	}
}

__global__ void specialAddKernel(float* C, const float* A, const float* B, int size, int subA2, int upperStrideA2, int upperA2, int subA1, int upperStrideA1, int betweenA, int subB2, int upperStrideB2, int upperB2, int subB1, int upperStrideB1, int betweenB)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < size)
	{
		int idxA = idx / upperStrideA2 % upperA2 * subA2 + idx / upperStrideA1 % betweenA * subA1 + idx % subA1;
		int idxB = idx / upperStrideB2 % upperB2 * subB2 + idx / upperStrideB1 % betweenB * subB1 + idx % subB1;

		C[idx] = A[idxA] + B[idxB];
	}
}

__global__ void universalAddKernel(float* C, const float* A, const float* B, int size, int dim, BroadcastStats stats)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < size)
	{
		int idxA = 0;
		int idxB = 0;

		for (int i = 0; i < dim; i++)
		{
			int coord = idx / stats.out_stride[i] % stats.out_shape[i];

			idxA += coord * stats.strideA[i];

			idxB += coord * stats.strideB[i];
		}

		C[idx] = A[idxA] + B[idxB];
	}
}

Tensor Tensor::operator+(const Tensor& B) const
{
	vector<int> newShape = dim() >= B.dim() ? shape : B.shape;

	if (newShape.size() > MAX_TENSOR_LENGTH)
		throw runtime_error("Tensor length exceeded!");

	int subDimsB, upperDimsB, upperStrideC_B;
	int subDimsA, upperDimsA, upperStrideC_A;

	BroadcastStats stats;

	int block = 256;

	if (dim() > B.dim())
	{
		vector<int> newB;
		newB.assign(dim() - B.dim(), 1);

		newB.insert(newB.end(), B.shape.begin(), B.shape.end());

		for (int i = 0; i < dim(); i++)
		{
			if (shape[i] != newB[i] && shape[i] != 1 && newB[i] != 1)
				throw runtime_error("Invalid Tensor shape!");

			if (shape[i] != 1)
				newShape[i] = shape[i];
			else if (newB[i] != 1)
				newShape[i] = newB[i];
			else
				newShape[i] = 1;
		}

		Tensor C(newShape);

		bool prev_stateA = 0;
		bool prev_stateB = 0;

		vector<int> regionsA;
		vector<int> regionsB;

		for (int i = 0; i < dim(); i++)
		{
			if (newShape[i] == 1)
				continue;

			bool current_stateA = shape[i] == 1;
			bool current_stateB = newB[i] == 1;

			if (current_stateA && !prev_stateA)
			{
				regionsA.push_back(i);
			}
			if (current_stateB && !prev_stateB)
			{
				regionsB.push_back(i);
			}

			prev_stateA = current_stateA;
			prev_stateB = current_stateB;
		}

		int grid = (C.total + block - 1) / block;

		bool supportA = regionsA.size() <= 1 || (regionsA.size() == 2 && regionsA[0] == 0);
		bool supportB = regionsB.size() <= 1 || (regionsB.size() == 2 && regionsB[0] == 0);

		if (supportA && supportB)
		{
			int axisA = -1;
			int axisB = -1;

			if (regionsA.size()) axisA = regionsA.size() == 1 ? regionsA[0] : regionsA[1];
			if (regionsB.size()) axisB = regionsB.size() == 1 ? regionsB[0] : regionsB[1];

			if (axisA == -1 && axisB == -1)
			{
				addKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, -1, -1, -1, -1, -1, -1);
			}
			else if (axisA == -1)
			{
				subDimsB = calculateStride(newB)[axisB];
				upperDimsB = calculateTotal(newB) / subDimsB;
				upperStrideC_B = axisB == 0 ? 1 : C.stride[axisB - 1];

				addKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, -1, subDimsB, -1, upperDimsB, -1, upperStrideC_B);
			}
			else if (axisB == -1)
			{
				subDimsA = stride[axisA];
				upperDimsA = total / subDimsA;
				upperStrideC_A = axisA == 0 ? 1 : C.stride[axisA - 1];

				addKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, subDimsA, -1, upperDimsA, -1, upperStrideC_A, -1);
			}
			else
			{
				subDimsB = calculateStride(newB)[axisB];
				upperDimsB = calculateTotal(newB) / subDimsB;
				upperStrideC_B = axisB == 0 ? 1 : C.stride[axisB - 1];

				subDimsA = stride[axisA];
				upperDimsA = total / subDimsA;
				upperStrideC_A = axisA == 0 ? 1 : C.stride[axisA - 1];

				addKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, subDimsA, subDimsB, upperDimsA, upperDimsB, upperStrideC_A, upperStrideC_B);
			}
		}
		else
		{
			vector<int> newStride;
			newStride.assign(dim() - B.dim(), 0);
			newStride.insert(newStride.end(), B.stride.begin(), B.stride.end());

			for (int i = 0; i < dim(); i++)
			{
				stats.strideB[i] = newB[i] != 1 ? newStride[i] : 0;
				stats.strideA[i] = shape[i] != 1 ? stride[i] : 0;
				stats.out_stride[i] = C.stride[i];
				stats.out_shape[i] = C.shape[i];
			}

			universalAddKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, C.dim(), stats);
		}

		return C;
	}
	else if (B.dim() > dim())
	{
		vector<int> newA;
		newA.assign(B.dim() - dim(), 1);

		newA.insert(newA.end(), shape.begin(), shape.end());

		for (int i = 0; i < B.dim(); i++)
		{
			if (newA[i] != B.shape[i] && newA[i] != 1 && B.shape[i] != 1)
				throw runtime_error("Invalid Tensor shape!");
				
			if (newA[i] != 1)
				newShape[i] = newA[i];
			else if (B.shape[i] != 1)
				newShape[i] = B.shape[i];
			else
				newShape[i] = 1;
		}

		Tensor C(newShape);

		int grid = (C.total + block - 1) / block;

		vector<int> regionsA;
		vector<int> regionsB;

		bool prev_stateA = 0;
		bool prev_stateB = 0;

		for (int i = 0; i < B.dim(); i++)
		{
			if (newShape[i] == 1)
				continue;

			bool current_stateA = newA[i] == 1;
			bool current_stateB = B.shape[i] == 1;

			if (current_stateA && !prev_stateA)
				regionsA.push_back(i);

			if (current_stateB && !prev_stateB)
				regionsB.push_back(i);

			prev_stateA = current_stateA;
			prev_stateB = current_stateB;
		}

		bool supportA = regionsA.size() <= 1 || (regionsA.size() == 2 && regionsA[0] == 0);
		bool supportB = regionsB.size() <= 1 || (regionsB.size() == 2 && regionsB[0] == 0);

		if (supportA && supportB)
		{
			int axisA = -1;
			int axisB = -1;

			if (regionsA.size()) axisA = regionsA.size() == 1 ? regionsA[0] : regionsA[1];
			if (regionsB.size()) axisB = regionsB.size() == 1 ? regionsB[0] : regionsB[1];

			if (axisA == -1 && axisB == -1)
			{
				addKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, -1, -1, -1, -1, -1, -1);
			}
			else if (axisA == -1)
			{
				subDimsB = B.stride[axisB];
				upperDimsB = B.total / subDimsB;
				upperStrideC_B = axisB == 0 ? 1 : C.stride[axisB - 1];

				addKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, -1, subDimsB, -1, upperDimsB, -1, upperStrideC_B);
			}
			else if (axisB == -1)
			{
				subDimsA = calculateStride(newA)[axisA];
				upperDimsA = calculateTotal(newA) / subDimsA;
				upperStrideC_A = axisA == 0 ? 1 : C.stride[axisA - 1];

				addKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, subDimsA, -1, upperDimsA, -1, upperStrideC_A, -1);
			}
			else
			{
				subDimsA = calculateStride(newA)[axisA];
				upperDimsA = calculateTotal(newA) / subDimsA;
				upperStrideC_A = axisA == 0 ? 1 : C.stride[axisA - 1];

				subDimsB = B.stride[axisB];
				upperDimsB = B.total / subDimsB;
				upperStrideC_B = axisB == 0 ? 1 : C.stride[axisB - 1];

				addKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, subDimsA, subDimsB, upperDimsA, upperDimsB, upperStrideC_A, upperStrideC_B);
			}
		}
		else
		{
			vector<int> newStride;
			newStride.assign(B.dim() - dim(), 0);
			newStride.insert(newStride.end(), stride.begin(), stride.end());

			for (int i = 0; i < B.dim(); i++)
			{
				stats.strideA[i] = newA[i] != 1 ? newStride[i] : 0;
				stats.strideB[i] = B.shape[i] != 1 ? B.stride[i] : 0;
				stats.out_stride[i] = C.stride[i];
				stats.out_shape[i] = C.shape[i];
			}

			universalAddKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, C.dim(), stats);
		}

		return C;
	}
	else
	{
		for (int i = 0; i < dim(); i++)
		{
			if (shape[i] != B.shape[i] && shape[i] != 1 && B.shape[i] != 1)
				throw runtime_error("Invalid Tensor shape!");
				
			if (shape[i] != 1)
				newShape[i] = shape[i];
			else if (B.shape[i] != 1)
				newShape[i] = B.shape[i];
			else
				newShape[i] = 1;
		}

		Tensor C(newShape);

		int grid = (C.total + block - 1) / block;

		bool prev_stateA = 0;
		bool prev_stateB = 0;

		vector<int> regionsA;
		vector<int> regionsB;

		for (int i = 0; i < dim(); i++)
		{
			if (newShape[i] == 1)
				continue;

			bool current_stateA = shape[i] == 1;
			bool current_stateB = B.shape[i] == 1;

			if (current_stateA && !prev_stateA)
				regionsA.push_back(i);

			if (current_stateB && !prev_stateB)
				regionsB.push_back(i);

			prev_stateA = current_stateA;
			prev_stateB = current_stateB;
		}

		bool supportA = regionsA.size() <= 1 || (regionsA.size() == 2 && regionsA[0] == 0);
		bool supportB = regionsB.size() <= 1 || (regionsB.size() == 2 && regionsB[0] == 0);

		if (supportA && supportB)
		{
			int axisA = -1;
			int axisB = -1;

			if (regionsA.size()) axisA = regionsA.size() == 1 ? regionsA[0] : regionsA[1];
			if (regionsB.size()) axisB = regionsB.size() == 1 ? regionsB[0] : regionsB[1];

			if (axisA == -1 && axisB == -1)
			{
				addKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, -1, -1, -1, -1, -1, -1);
			}
			else if (axisA == -1)
			{
				subDimsB = B.stride[axisB];
				upperDimsB = B.total / subDimsB;
				upperStrideC_B = axisB == 0 ? 1 : C.stride[axisB - 1];

				addKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, -1, subDimsB, -1, upperDimsB, -1, upperStrideC_B);
			}
			else if (axisB == -1)
			{
				subDimsA = stride[axisA];
				upperDimsA = total / subDimsA;
				upperStrideC_A = axisA == 0 ? 1 : C.stride[axisA - 1];

				addKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, subDimsA, -1, upperDimsA, -1, upperStrideC_A, -1);
			}
			else
			{
				subDimsA = stride[axisA];
				upperDimsA = total / subDimsA;
				upperStrideC_A = axisA == 0 ? 1 : C.stride[axisA - 1];

				subDimsB = B.stride[axisB];
				upperDimsB = B.total / subDimsB;
				upperStrideC_B = axisB == 0 ? 1 : C.stride[axisB - 1];

				addKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, subDimsA, subDimsB, upperDimsA, upperDimsB, upperStrideC_A, upperStrideC_B);
			}
		}
		else
		{
			for (int i = 0; i < dim(); i++)
			{
				stats.strideA[i] = shape[i] != 1 ? stride[i] : 0;
				stats.strideB[i] = B.shape[i] != 1 ? B.stride[i] : 0;
				stats.out_stride[i] = C.stride[i];
				stats.out_shape[i] = C.shape[i];
			}

			universalAddKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, C.dim(), stats);
		}

		return C;
	}
}

__global__ void subKernel(float* C, const float* A, const float* B, int size, int subA, int subB, int upperA, int upperB, int strideA, int strideB)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < size)
	{
		int idxA = subA == -1 ? idx : idx / strideA % upperA * subA + idx % subA;
		int idxB = subB == -1 ? idx : idx / strideB % upperB * subB + idx % subB;

		C[idx] = A[idxA] - B[idxB];
	}
}

__global__ void universalSubKernel(float* C, const float* A, const float* B, int size, int dim, BroadcastStats stats)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < size)
	{
		int idxA = 0;
		int idxB = 0;

		for (int i = 0; i < dim; i++)
		{
			int coord = idx / stats.out_stride[i] % stats.out_shape[i];

			idxA += coord * stats.strideA[i];

			idxB += coord * stats.strideB[i];
		}

		C[idx] = A[idxA] - B[idxB];
	}
}

Tensor Tensor::operator-(const Tensor& B) const
{
	vector<int> newShape = dim() >= B.dim() ? shape : B.shape;

	if (newShape.size() > MAX_TENSOR_LENGTH)
		throw runtime_error("Tensor length exceeded!");

	int subDimsB, upperDimsB, upperStrideC_B;
	int subDimsA, upperDimsA, upperStrideC_A;

	BroadcastStats stats;

	int block = 256;

	if (dim() > B.dim())
	{
		vector<int> newB;
		newB.assign(dim() - B.dim(), 1);

		newB.insert(newB.end(), B.shape.begin(), B.shape.end());

		for (int i = 0; i < dim(); i++)
		{
			if (shape[i] != newB[i] && shape[i] != 1 && newB[i] != 1)
				throw runtime_error("Invalid Tensor shape!");

			if (shape[i] != 1)
				newShape[i] = shape[i];
			else if (newB[i] != 1)
				newShape[i] = newB[i];
			else
				newShape[i] = 1;
		}

		Tensor C(newShape);

		bool prev_stateA = 0;
		bool prev_stateB = 0;

		vector<int> regionsA;
		vector<int> regionsB;

		for (int i = 0; i < dim(); i++)
		{
			if (newShape[i] == 1)
				continue;

			bool current_stateA = shape[i] == 1;
			bool current_stateB = newB[i] == 1;

			if (current_stateA && !prev_stateA)
			{
				regionsA.push_back(i);
			}
			if (current_stateB && !prev_stateB)
			{
				regionsB.push_back(i);
			}

			prev_stateA = current_stateA;
			prev_stateB = current_stateB;
		}

		int grid = (C.total + block - 1) / block;

		bool supportA = regionsA.size() <= 1 || (regionsA.size() == 2 && regionsA[0] == 0);
		bool supportB = regionsB.size() <= 1 || (regionsB.size() == 2 && regionsB[0] == 0);

		if (supportA && supportB)
		{
			int axisA = -1;
			int axisB = -1;

			if (regionsA.size()) axisA = regionsA.size() == 1 ? regionsA[0] : regionsA[1];
			if (regionsB.size()) axisB = regionsB.size() == 1 ? regionsB[0] : regionsB[1];

			if (axisA == -1 && axisB == -1)
			{
				subKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, -1, -1, -1, -1, -1, -1);
			}
			else if (axisA == -1)
			{
				subDimsB = calculateStride(newB)[axisB];
				upperDimsB = calculateTotal(newB) / subDimsB;
				upperStrideC_B = axisB == 0 ? 1 : C.stride[axisB - 1];

				subKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, -1, subDimsB, -1, upperDimsB, -1, upperStrideC_B);
			}
			else if (axisB == -1)
			{
				subDimsA = stride[axisA];
				upperDimsA = total / subDimsA;
				upperStrideC_A = axisA == 0 ? 1 : C.stride[axisA - 1];

				subKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, subDimsA, -1, upperDimsA, -1, upperStrideC_A, -1);
			}
			else
			{
				subDimsB = calculateStride(newB)[axisB];
				upperDimsB = calculateTotal(newB) / subDimsB;
				upperStrideC_B = axisB == 0 ? 1 : C.stride[axisB - 1];

				subDimsA = stride[axisA];
				upperDimsA = total / subDimsA;
				upperStrideC_A = axisA == 0 ? 1 : C.stride[axisA - 1];

				subKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, subDimsA, subDimsB, upperDimsA, upperDimsB, upperStrideC_A, upperStrideC_B);
			}
		}
		else
		{
			vector<int> newStride;
			newStride.assign(dim() - B.dim(), 0);
			newStride.insert(newStride.end(), B.stride.begin(), B.stride.end());

			for (int i = 0; i < dim(); i++)
			{
				stats.strideB[i] = newB[i] != 1 ? newStride[i] : 0;
				stats.strideA[i] = shape[i] != 1 ? stride[i] : 0;
				stats.out_stride[i] = C.stride[i];
				stats.out_shape[i] = C.shape[i];
			}

			universalSubKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, C.dim(), stats);
		}

		return C;
	}
	else if (B.dim() > dim())
	{
		vector<int> newA;
		newA.assign(B.dim() - dim(), 1);

		newA.insert(newA.end(), shape.begin(), shape.end());

		for (int i = 0; i < B.dim(); i++)
		{
			if (newA[i] != B.shape[i] && newA[i] != 1 && B.shape[i] != 1)
				throw runtime_error("Invalid Tensor shape!");

			if (newA[i] != 1)
				newShape[i] = newA[i];
			else if (B.shape[i] != 1)
				newShape[i] = B.shape[i];
			else
				newShape[i] = 1;
		}

		Tensor C(newShape);

		int grid = (C.total + block - 1) / block;

		vector<int> regionsA;
		vector<int> regionsB;

		bool prev_stateA = 0;
		bool prev_stateB = 0;

		for (int i = 0; i < B.dim(); i++)
		{
			if (newShape[i] == 1)
				continue;

			bool current_stateA = newA[i] == 1;
			bool current_stateB = B.shape[i] == 1;

			if (current_stateA && !prev_stateA)
				regionsA.push_back(i);

			if (current_stateB && !prev_stateB)
				regionsB.push_back(i);

			prev_stateA = current_stateA;
			prev_stateB = current_stateB;
		}

		bool supportA = regionsA.size() <= 1 || (regionsA.size() == 2 && regionsA[0] == 0);
		bool supportB = regionsB.size() <= 1 || (regionsB.size() == 2 && regionsB[0] == 0);

		if (supportA && supportB)
		{
			int axisA = -1;
			int axisB = -1;

			if (regionsA.size()) axisA = regionsA.size() == 1 ? regionsA[0] : regionsA[1];
			if (regionsB.size()) axisB = regionsB.size() == 1 ? regionsB[0] : regionsB[1];

			if (axisA == -1 && axisB == -1)
			{
				subKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, -1, -1, -1, -1, -1, -1);
			}
			else if (axisA == -1)
			{
				subDimsB = B.stride[axisB];
				upperDimsB = B.total / subDimsB;
				upperStrideC_B = axisB == 0 ? 1 : C.stride[axisB - 1];

				subKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, -1, subDimsB, -1, upperDimsB, -1, upperStrideC_B);
			}
			else if (axisB == -1)
			{
				subDimsA = calculateStride(newA)[axisA];
				upperDimsA = calculateTotal(newA) / subDimsA;
				upperStrideC_A = axisA == 0 ? 1 : C.stride[axisA - 1];

				subKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, subDimsA, -1, upperDimsA, -1, upperStrideC_A, -1);
			}
			else
			{
				subDimsA = calculateStride(newA)[axisA];
				upperDimsA = calculateTotal(newA) / subDimsA;
				upperStrideC_A = axisA == 0 ? 1 : C.stride[axisA - 1];

				subDimsB = B.stride[axisB];
				upperDimsB = B.total / subDimsB;
				upperStrideC_B = axisB == 0 ? 1 : C.stride[axisB - 1];

				subKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, subDimsA, subDimsB, upperDimsA, upperDimsB, upperStrideC_A, upperStrideC_B);
			}
		}
		else
		{
			vector<int> newStride;
			newStride.assign(B.dim() - dim(), 0);
			newStride.insert(newStride.end(), stride.begin(), stride.end());

			for (int i = 0; i < B.dim(); i++)
			{
				stats.strideA[i] = newA[i] != 1 ? newStride[i] : 0;
				stats.strideB[i] = B.shape[i] != 1 ? B.stride[i] : 0;
				stats.out_stride[i] = C.stride[i];
				stats.out_shape[i] = C.shape[i];
			}

			universalSubKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, C.dim(), stats);
		}

		return C;
	}
	else
	{
		for (int i = 0; i < dim(); i++)
		{
			if (shape[i] != B.shape[i] && shape[i] != 1 && B.shape[i] != 1)
				throw runtime_error("Invalid Tensor shape!");

			if (shape[i] != 1)
				newShape[i] = shape[i];
			else if (B.shape[i] != 1)
				newShape[i] = B.shape[i];
			else
				newShape[i] = 1;
		}

		Tensor C(newShape);

		int grid = (C.total + block - 1) / block;

		bool prev_stateA = 0;
		bool prev_stateB = 0;

		vector<int> regionsA;
		vector<int> regionsB;

		for (int i = 0; i < dim(); i++)
		{
			if (newShape[i] == 1)
				continue;

			bool current_stateA = shape[i] == 1;
			bool current_stateB = B.shape[i] == 1;

			if (current_stateA && !prev_stateA)
				regionsA.push_back(i);

			if (current_stateB && !prev_stateB)
				regionsB.push_back(i);

			prev_stateA = current_stateA;
			prev_stateB = current_stateB;
		}

		bool supportA = regionsA.size() <= 1 || (regionsA.size() == 2 && regionsA[0] == 0);
		bool supportB = regionsB.size() <= 1 || (regionsB.size() == 2 && regionsB[0] == 0);

		if (supportA && supportB)
		{
			int axisA = -1;
			int axisB = -1;

			if (regionsA.size()) axisA = regionsA.size() == 1 ? regionsA[0] : regionsA[1];
			if (regionsB.size()) axisB = regionsB.size() == 1 ? regionsB[0] : regionsB[1];

			if (axisA == -1 && axisB == -1)
			{
				subKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, -1, -1, -1, -1, -1, -1);
			}
			else if (axisA == -1)
			{
				subDimsB = B.stride[axisB];
				upperDimsB = B.total / subDimsB;
				upperStrideC_B = axisB == 0 ? 1 : C.stride[axisB - 1];

				subKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, -1, subDimsB, -1, upperDimsB, -1, upperStrideC_B);
			}
			else if (axisB == -1)
			{
				subDimsA = stride[axisA];
				upperDimsA = total / subDimsA;
				upperStrideC_A = axisA == 0 ? 1 : C.stride[axisA - 1];

				subKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, subDimsA, -1, upperDimsA, -1, upperStrideC_A, -1);
			}
			else
			{
				subDimsA = stride[axisA];
				upperDimsA = total / subDimsA;
				upperStrideC_A = axisA == 0 ? 1 : C.stride[axisA - 1];

				subDimsB = B.stride[axisB];
				upperDimsB = B.total / subDimsB;
				upperStrideC_B = axisB == 0 ? 1 : C.stride[axisB - 1];

				subKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, subDimsA, subDimsB, upperDimsA, upperDimsB, upperStrideC_A, upperStrideC_B);
			}
		}
		else
		{
			for (int i = 0; i < dim(); i++)
			{
				stats.strideA[i] = shape[i] != 1 ? stride[i] : 0;
				stats.strideB[i] = B.shape[i] != 1 ? B.stride[i] : 0;
				stats.out_stride[i] = C.stride[i];
				stats.out_shape[i] = C.shape[i];
			}

			universalSubKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, C.dim(), stats);
		}

		return C;
	}
}

__global__ void mulScalarKernel(const float* A, float* C, float x, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = A[idx] * x;
	}
}

Tensor Tensor::operator*(float x) const
{
	Tensor C(shape);

	int block = 256;
	int grid = (total + block - 1) / block;

	mulScalarKernel << <grid, block >> > (getFloatData(), C.getFloatData(), x, total);

	return C;
}

__global__ void divScalarKernel(const float* A, float* C, float x, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = A[idx] / x;
	}
}

Tensor Tensor::operator/(float x) const
{
	if (x == 0)
		throw runtime_error("ivision by zero not allowed!");

	Tensor C(shape);

	int block = 256;
	int grid = (total + block - 1) / block;

	divScalarKernel << <grid, block >> > (getFloatData(), C.getFloatData(), x, total);

	return C;
}

__global__ void addScalarKernel(const float* A, float* C, float x, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = A[idx] + x;
	}
}

Tensor Tensor::operator+(float x) const
{
	Tensor C(shape);

	int block = 256;
	int grid = (total + block - 1) / block;

	addScalarKernel << <grid, block >> > (getFloatData(), C.getFloatData(), x, total);

	return C;
}

__global__ void subScalarKernel(const float* A, float* C, float x, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = A[idx] - x;
	}
}

Tensor Tensor::operator-(float x) const
{
	Tensor C(shape);

	int block = 256;
	int grid = (total + block - 1) / block;

	subScalarKernel << <grid, block >> > (getFloatData(), C.getFloatData(), x, total);

	return C;
}

Tensor operator*(float x, const Tensor& A)
{
	return A * x;
}

Tensor operator+(float x, const Tensor& A)
{
	return A + x;
}

__global__ void scalarDivKernel(const float* A, float* C, float x, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = x / A[idx];
	}
}

Tensor operator/(float x, const Tensor& A)
{
	Tensor C(A.getShape());

	int block = 256;
	int grid = (A.size() + block - 1) / block;

	scalarDivKernel << <grid, block >> > (A.getFloatData(), C.getFloatData(), x, A.size());

	return C;
}

__global__ void scalarSubKernel(const float* A, float* C, float x, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = x - A[idx];
	}
}

Tensor operator-(float x, const Tensor& A)
{
	Tensor C(A.getShape());

	int block = 256;
	int grid = (A.size() + block - 1) / block;

	scalarSubKernel << <grid, block >> > (A.getFloatData(), C.getFloatData(), x, A.size());

	return C;
}

__global__ void equalKernel(const float* A, const float* B, float* C, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = (A[idx] == B[idx]);
	}
}

Tensor Tensor::operator==(const Tensor& B) const
{
	if (shape != B.shape)
		throw runtime_error("Shapes do not match!");

	Tensor C(shape);

	int block = 256;
	int grid = (total + block - 1) / block;

	equalKernel << <grid, block >> > (getFloatData(), B.getFloatData(), C.getFloatData(), total);

	return C;
}

__global__ void greaterthKernel(const float* A, float* C, float x, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = (A[idx] > x);
	}
}

Tensor Tensor::operator>(float x) const
{
	Tensor C(shape);

	int block = 256;
	int grid = (total + block - 1) / block;

	greaterthKernel << <grid, block >> > (getFloatData(), C.getFloatData(), x, total);

	return C;
}

__global__ void lessthKernel(const float* A, float* C, float x, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = (A[idx] < x);
	}
}

Tensor Tensor::operator<(float x) const
{
	Tensor C(shape);

	int block = 256;
	int grid = (total + block - 1) / block;

	lessthKernel << <grid, block >> > (getFloatData(), C.getFloatData(), x, total);

	return C;
}

__global__ void matmulKernel(float* C, const float* A, const float* B, int size, int M, int K, int N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < size)
	{
		int upperIdxA = idx / N;
		int upperIdxB = idx / (N * M);
		int subIdxB = idx % N;

		int idxA = upperIdxA * K;
		int idxB = upperIdxB * K * N + subIdxB;

		float sum = 0.0f;

		for (int i = 0; i < K; i++)
		{
			sum += A[idxA + i] * B[idxB + i * N];
		}

		C[idx] = sum;
	}
}

Tensor Tensor::matmul(const Tensor& B) const
{
	if (dim() < 2)
		throw runtime_error("Tensors must be at least 2 dimesnional!");

	int K = shape[dim() - 1];

	if (K != B.shape[B.dim() - 2])
		throw runtime_error("Invalid dimesnions for matrix multiplication!");

	vector<int> batchA = shape;
	vector<int> batchB = B.shape;

	batchA.resize(dim() - 2);
	batchB.resize(B.dim() - 2);

	if (batchA != batchB)
		throw runtime_error("Batch size must be equal!");

	int M = shape[dim() - 2];

	int N = B.shape[B.dim() - 1];

	vector<int> newShape = shape;

	newShape.pop_back();
	newShape.push_back(B.shape[B.dim() - 1]);

	Tensor C(newShape);

	int block = 256;
	int grid = (C.total + block - 1) / block;

	matmulKernel << <grid, block >> > (C.getFloatData(), getFloatData(), B.getFloatData(), C.total, M, K, N);

	return C;
}

__global__ void TKernel(float* C, const float* A, int size, int N, int M)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < size)
	{
		int idxC = idx % M * N + idx / M + idx / (N * M) * ((N * M) - N);
		C[idxC] = A[idx];
	}
}

Tensor Tensor::T() const
{
	if (dim() < 2)
		throw runtime_error("Tensor must be at least 2 dimesnional!");

	vector<int> newShape = shape;

	newShape.resize(dim() - 2);
	newShape.push_back(shape[dim() - 1]);
	newShape.push_back(shape[dim() - 2]);

	Tensor C(newShape);

	int block = 256;
	int grid = (total + block - 1) / block;

	int N = shape[dim() - 2];
	int M = shape[dim() - 1];

	TKernel << <grid, block >> > (C.getFloatData(), getFloatData(), total, N, M);

	return C;
}

__global__ void theSumKernel(float* C, const float* A, int size, int subDims, int redDim)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		float sum = 0.0f;

		int upperIdx = idx / subDims;

		int subIdx = idx % subDims;

		int a_idx = upperIdx * subDims * redDim + subIdx;

		for (int i = 0; i < redDim; i++)
		{
			sum += A[a_idx + i * subDims];
		}

		C[idx] = sum;
	}
}

__global__ void sumallKernel(float* C, const float* A, int size)
{
	extern __shared__ float sh[];

	int sh_idx = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	float val = 0.0f;

	if (idx < size)
		val = A[idx];

	sh[sh_idx] = val;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
	{
		if (sh_idx < stride)
			sh[sh_idx] += sh[sh_idx + stride];

		__syncthreads();
	}

	if (sh_idx == 0)
		atomicAdd(C, sh[0]);
}

Tensor Tensor::sum(const Tensor& A, int axis, bool keepdim)
{
	if (axis<-1 || axis>A.dim()-1)
		throw runtime_error("Invalid axis value. Please input -1 or a valid value!");

	int block = 256;

	if (axis != -1)
	{
		vector<int> newShape = A.shape;

		if (keepdim)
			newShape[axis] = 1;
		else
		{
			newShape.erase(newShape.begin() + axis);

			if (newShape.empty())
				newShape.push_back(1);
		}

		Tensor C(newShape);

		int subDims = A.stride[axis];

		int redDim = A.shape[axis];

		int grid = (C.total + block - 1) / block;

		theSumKernel << <grid, block >> > (C.getFloatData(), A.getFloatData(), C.total, subDims, redDim);

		cudaError_t error = cudaGetLastError();

		if (error != cudaSuccess)
			throw runtime_error(cudaGetErrorString(error));

		return C;
	}
	else
	{
		Tensor C({ 1 });

		int grid = (A.total + block - 1) / block;

		cudaMemset(C.getFloatData(), 0, sizeof(float));

		sumallKernel << <grid, block, block * sizeof(float) >> > (C.getFloatData(), A.getFloatData(), A.total);

		cudaError_t error = cudaGetLastError();

		if (error != cudaSuccess)
			throw runtime_error(cudaGetErrorString(error));

		return C;
	}
}

__global__ void theArgmaxKernel(float* C, const float* A, int size, int subDims, int redDim)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		float the_max = -FLT_MAX;
		int the_argmax = 0;

		int upperIdx = idx / subDims;

		int subIdx = idx % subDims;

		int a_idx = upperIdx * subDims * redDim + subIdx;

		for (int i = 0; i < redDim; i++)
		{
			if (A[a_idx + i * subDims] > the_max)
			{
				the_max = A[a_idx + i * subDims];
				the_argmax = i;
			}
		}

		C[idx] = the_argmax;
	}
}

__global__ void argmaxAllStartKernel(MaxStats* C, const float* A, int size)
{
	extern __shared__ MaxStats shm[];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int sh_idx = threadIdx.x;

	if (idx < size)
	{
		shm[sh_idx].value = A[idx];
		shm[sh_idx].index = idx;
	}
	else
	{
		shm[sh_idx].value = -FLT_MAX;
		shm[sh_idx].index = -1;
	}
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
	{
		if (sh_idx<stride)
		{
			if (shm[sh_idx + stride].value > shm[sh_idx].value || (shm[sh_idx + stride].value == shm[sh_idx].value && shm[sh_idx + stride].index >= 0 && shm[sh_idx + stride].index < shm[sh_idx].index))
			{
				shm[sh_idx] = shm[sh_idx + stride];
			}
		}

		__syncthreads();
	}

	if (sh_idx == 0)
	{
		C[blockIdx.x] = shm[0];
	}
}

__global__ void argmaxAllNextKernel(MaxStats* C, const MaxStats* A, int size)
{
	extern __shared__ MaxStats shm[];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int sh_idx = threadIdx.x;

	if (idx < size)
	{
		shm[sh_idx] = A[idx];
	}
	else
	{
		shm[sh_idx].value = -FLT_MAX;
		shm[sh_idx].index = -1;
	}

	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
	{
		if (sh_idx < stride)
		{
			if (shm[sh_idx + stride].value > shm[sh_idx].value || (shm[sh_idx + stride].value == shm[sh_idx].value && shm[sh_idx + stride].index >= 0 && shm[sh_idx+stride].index < shm[sh_idx].index))
			{
				shm[sh_idx] = shm[sh_idx + stride];
			}
		}

		__syncthreads();
	}

	if (sh_idx == 0)
	{
		C[blockIdx.x] = shm[0];
	}
}

__global__ void resultConversionKernel(float* C, const MaxStats* A)
{
	if (blockIdx.x == 0 && threadIdx.x == 0)
	{
		C[0] = A[0].index;
	}
}

Tensor Tensor::argmax(const Tensor& A, int axis, bool keepdim)
{
	if (axis<-1 || axis>A.dim()-1)
		throw runtime_error("Invalid axis value. Please input -1 or a valid value!");

	int block = 256;

	if (axis == -1)
	{
		int grid = (A.total + block - 1) / block;

		MaxStats* C;

		cudaMalloc(&C, grid * sizeof(MaxStats));

		argmaxAllStartKernel << <grid, block, block * sizeof(MaxStats) >> > (C, A.getFloatData(), A.total);

		cudaError_t error = cudaGetLastError();

		if (error != cudaSuccess)
			throw runtime_error(cudaGetErrorString(error));

		while (grid > 1)
		{
			int newGrid = (grid + block - 1) / block;

			MaxStats* partial;

			cudaMalloc(&partial, newGrid * sizeof(MaxStats));

			argmaxAllNextKernel << <newGrid, block, block * sizeof(MaxStats) >> > (partial, C, grid);

			error = cudaGetLastError();

			if (error != cudaSuccess)
			{
				cudaFree(partial);
				cudaFree(C);
				throw runtime_error(cudaGetErrorString(error));
			}

			grid = newGrid;

			cudaFree(C);

			C = partial;
		}

		Tensor res({ 1 });

		resultConversionKernel << <1, 1 >> > (res.getFloatData(), C);

		error = cudaGetLastError();

		if (error != cudaSuccess)
		{
			cudaFree(C);
			throw runtime_error(cudaGetErrorString(error));
		}

		cudaFree(C);

		return res;
	}

	vector<int> newShape = A.shape;

	if (keepdim)
		newShape[axis] = 1;
	else
	{
		newShape.erase(newShape.begin() + axis);

		if (newShape.empty())
			newShape.push_back(1);
	}

	Tensor C(newShape);

	int subDims = A.stride[axis];

	int redDim = A.shape[axis];

	int grid = (C.total + block - 1) / block;

	theArgmaxKernel << <grid, block >> > (C.getFloatData(), A.getFloatData(), C.total, subDims, redDim);

	cudaError_t error = cudaGetLastError();

	if (error != cudaSuccess)
		throw runtime_error(cudaGetErrorString(error));

	return C;
}

__global__ void maxallKernel(float* C, const float* A, int size)
{
	int sh_idx = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ float sh[];

	float val = -FLT_MAX;

	if (idx < size)
		val = A[idx];

	sh[sh_idx] = val;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
	{
		if (sh_idx<stride && sh[sh_idx + stride]>sh[sh_idx])
			sh[sh_idx] = sh[sh_idx + stride];

		__syncthreads();
	}

	if (sh_idx == 0)
		C[blockIdx.x] = sh[0];
}

__global__ void theMaxKernel(float* C, const float* A, int size, int subDims, int redDim)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		float the_max = -FLT_MAX;
		int upperIdx = idx / subDims;
		int subIdx = idx % subDims;
		int a_idx = upperIdx * subDims * redDim + subIdx;

		for (int i = 0; i < redDim; i++)
		{
			if (A[a_idx + i * subDims] > the_max)
				the_max = A[a_idx + i * subDims];
		}

		C[idx] = the_max;
	}
}

Tensor Tensor::maxT(const Tensor& A, int axis, bool keepdim)
{
	if (axis<-1 || axis>A.dim()-1)
		throw runtime_error("Axis value outside of tensor dimension! Please input -1 or a valid value!");

	int block = 256;

	if (axis != -1)
	{
		vector<int> newShape = A.shape;

		if (keepdim)
			newShape[axis] = 1;
		else
		{
			newShape.erase(newShape.begin() + axis);

			if (newShape.empty())
				newShape.push_back(1);
		}

		Tensor C(newShape);

		int subDims = A.stride[axis];

		int redDim = A.shape[axis];

		int grid = (C.total + block - 1) / block;

		theMaxKernel << <grid, block >> > (C.getFloatData(), A.getFloatData(), C.total, subDims, redDim);

		cudaError_t error = cudaGetLastError();

		if (error != cudaSuccess)
			throw runtime_error(cudaGetErrorString(error));

		return C;
	}
	else
	{
		int grid = (A.total + block - 1) / block;

		Tensor C({ grid });

		maxallKernel << <grid, block, block * sizeof(float) >> > (C.getFloatData(), A.getFloatData(), A.total);

		cudaError_t error = cudaGetLastError();

		if (error != cudaSuccess)
			throw runtime_error(cudaGetErrorString(error));

		while (grid > 1)
		{
			int newGrid = (grid + block - 1) / block;

			Tensor partial({ newGrid });

			maxallKernel << <newGrid, block, block * sizeof(float) >> > (partial.getFloatData(), C.getFloatData(), C.total);

			cudaError_t error = cudaGetLastError();

			if (error != cudaSuccess)
				throw runtime_error(cudaGetErrorString(error));

			grid = newGrid;

			C = move(partial);
		}

		return C;
	}
}

__global__ void powKernel(const float* A, float* C, float p, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = powf(A[idx], p);
	}
}

Tensor Tensor::powT(const Tensor& A, float power)
{
	Tensor C(A.shape);

	int block = 256;
	int grid = (A.size() + block - 1) / block;

	powKernel << <grid, block >> > (A.getFloatData(), C.getFloatData(), power, A.total);

	return C;
}

__global__ void sqrtKernel(const float* A, float* C, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = sqrt(A[idx]);
	}
}

Tensor Tensor::sqrtT(const Tensor& A)
{
	Tensor C(A.shape);

	int block = 256;
	int grid = (A.size() + block - 1) / block;

	sqrtKernel << <grid, block >> > (A.getFloatData(), C.getFloatData(), A.total);

	return C;
}

__global__ void expKernel(const float* A, float* C, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = exp(A[idx]);
	}
}

Tensor Tensor::expT(const Tensor& A)
{
	Tensor C(A.shape);

	int block = 256;
	int grid = (A.size() + block - 1) / block;

	expKernel << <grid, block >> > (A.getFloatData(), C.getFloatData(), A.total);

	return C;
}

__global__ void logKernel(const float* A, float* C, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = log(A[idx]);
	}
}

Tensor Tensor::logT(const Tensor& A)
{
	Tensor C(A.shape);

	int block = 256;
	int grid = (A.size() + block - 1) / block;

	logKernel << <grid, block >> > (A.getFloatData(), C.getFloatData(), A.total);

	return C;
}

__global__ void absKernel(const float* A, float* C, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = abs(A[idx]);
	}
}

Tensor Tensor::absT(const Tensor& A)
{
	Tensor C(A.shape);

	int block = 256;
	int grid = (A.size() + block - 1) / block;

	absKernel << <grid, block >> > (A.getFloatData(), C.getFloatData(), A.total);

	return C;
}

__global__ void clipKernel(const float* A, float* C, float minVal, float maxVal, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		if (A[idx] < maxVal)
		{
			if (A[idx] > minVal)
			{
				C[idx] = A[idx];
			}
			else
				C[idx] = minVal;
		}
		else
			C[idx] = maxVal;
	}
}

Tensor Tensor::clipT(const Tensor& A, float minVal, float maxVal)
{
	if (minVal > maxVal)
		throw runtime_error("Min value cannot be bigger than max value!");

	Tensor C(A.shape);

	int block = 256;
	int grid = (A.size() + block - 1) / block;

	clipKernel << <grid, block >> > (A.getFloatData(), C.getFloatData(), minVal, maxVal, A.total);

	return C;
}

__global__ void tanhKernel(const float* A, float* C, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = tanh(A[idx]);
	}
}

Tensor Tensor::tanhT(const Tensor& A)
{
	Tensor C(A.shape);

	int block = 256;
	int grid = (A.total + block - 1) / block;

	tanhKernel << <grid, block >> > (A.getFloatData(), C.getFloatData(), A.total);

	return C;
}

__global__ void reluKernel(const float* A, float* C, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = A[idx] > 0.0f ? A[idx] : 0.0f;
	}
}

Tensor Tensor::relu(const Tensor& A)
{
	Tensor C(A.shape);

	int block = 256;
	int grid = (A.size() + block - 1) / block;

	reluKernel << <grid, block >> > (A.getFloatData(), C.getFloatData(), A.total);

	return C;
}

__global__ void der_reluKernel(const float* A, float* C, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = A[idx] > 0.0f ? 1.0f : 0.0f;
	}
}

Tensor Tensor::der_relu(const Tensor& A)
{
	Tensor C(A.shape);

	int block = 256;
	int grid = (A.size() + block - 1) / block;

	der_reluKernel << <grid, block >> > (A.getFloatData(), C.getFloatData(), A.total);

	return C;
}

Tensor Tensor::clone() const
{
	Tensor C(shape, dtype);

	cudaMemcpy(C.data, data, total * calcDataSize(dtype), cudaMemcpyDeviceToDevice);

	return C;
}

float Tensor::toScalar() const
{
	if (total != 1)
		throw runtime_error("Tensor must contain a single element!");

	float res = 0.0f;

	cudaMemcpy(&res, data, total * sizeof(float), cudaMemcpyDeviceToHost);

	return res;
}