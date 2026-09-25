#include "CPUTensor.h";
#include "Tensor.cuh"
#include <stdexcept>
#include <iostream>
#include <random>
#include <fstream>

static random_device rd;
static mt19937 gen(rd());

CPUTensor::CPUTensor(): data(), shape(), stride(), total(0), dtype(DataType::float32) {}

int CPUTensor::calculateTotal(const vector<int>& shape)
{
	int res = 1;

	for (int i = 0; i < static_cast<int>(shape.size()); i++)
	{
		if (shape[i] <= 0)
			throw runtime_error("Tensor dimension cannot be less than or 0.");
		res *= shape[i];
	}

	return res;
}

vector<int> CPUTensor::calculateStride(const vector<int>& shape)
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

template<typename T>
void CPUTensor::recursivePrint(const vector<T>& vals, int dim, int offset, int indent) const
{
	int rank = static_cast<int>(shape.size());

	if (rank == 0)
	{
		cout << vals[0];
		return;
	}

	if (dim == rank - 1)
	{
		cout << "[";
		for (int i = 0; i < shape[dim]; i++)
		{
			cout << vals[offset + i * stride[dim]];

			if (i != shape[dim] - 1)
				cout << ", ";
		}
		cout << "]";
		return;
	}

	cout << "[";
	for (int i = 0; i < shape[dim]; i++)
	{
		int new_offset = offset + i * stride[dim];

		if (i > 0)
		{
			cout << ",\n";
			cout << string(indent + 1, ' ');
		}

		recursivePrint(vals, dim + 1, new_offset, indent + 1);
	}
	cout << "]";
}

CPUTensor::CPUTensor(const vector<int>& shape, DataType dtype) : data(), shape(shape), stride(calculateStride(shape)), total(0), dtype(dtype)
{
	total = calculateTotal(shape);
	switch (dtype)
	{
	case DataType::float32:
		data.f = vector<float>(total);
		break;
	case DataType::int32:
		data.i = vector<int32_t>(total);
		break;
	default:
		throw runtime_error("Invalid data type!");
	}
}

float& CPUTensor::operator()(int index)
{
	return data.f[index];
}

const float& CPUTensor::operator()(int index) const
{
	return data.f[index];
}

float* CPUTensor::getFloatData()
{
	if (dtype != DataType::float32)
		throw runtime_error("Data type must be float!");

	return data.f.data();
}

const float* CPUTensor::getFloatData() const
{
	if (dtype != DataType::float32)
		throw runtime_error("Data type must be float!");

	return data.f.data();
}

int* CPUTensor::getIntData()
{
	if (dtype != DataType::int32)
		throw runtime_error("Data type must be int!");

	return data.i.data();
}
const int* CPUTensor::getIntData() const
{
	if (dtype != DataType::int32)
		throw runtime_error("Data type must be int!");

	return data.i.data();
}

CPUTensor& CPUTensor::operator=(const vector<float>& X)
{
	if (X.size() != total)
		throw runtime_error("Sizes dont match!");
	data.f = X;

	return *this;
}

int CPUTensor::size() const
{
	return total;
}

int CPUTensor::dim() const
{
	return static_cast<int>(shape.size());
}

const vector<int>& CPUTensor::getShape() const
{
	return shape;
}

const vector<int>& CPUTensor::getStride() const
{
	return stride;
}

DataType CPUTensor::getDataType() const
{
	return dtype;
}

Tensor CPUTensor::toCUDA() const
{
	Tensor T(shape, dtype);

	switch (dtype)
	{
	case DataType::float32:
		cudaMemcpy(T.getFloatData(), getFloatData(), T.byteSize(), cudaMemcpyHostToDevice);
		break;
	case DataType::int32:
		cudaMemcpy(T.getIntData(), getIntData(), T.byteSize(), cudaMemcpyHostToDevice);
		break;
	default:
		throw runtime_error("Invalid data type!");
	}

	return T;
}

void CPUTensor::print() const
{
	switch (dtype)
	{
	case DataType::float32:
		recursivePrint(data.f, 0, 0, 0);
		break;
	case DataType::int32:
		recursivePrint(data.i, 0, 0, 0);
		break;
	default:
		throw runtime_error("invalid data type!");
	}

	cout << "\n";
}

void CPUTensor::print_dims() const
{
	cout << "\n(";
	for (int i = 0; i < dim(); i++)
	{
		cout << shape[i];
		if (i != dim() - 1)
			cout << ", ";
	}
	cout << ")";
}

void CPUTensor::setSeed(int seed)
{
	gen.seed(seed);
}

CPUTensor CPUTensor::random(const vector<int>& shape)
{
	normal_distribution<float> dist(0.0f, 1.0f);

	CPUTensor T(shape);

	for (int i = 0; i < T.size(); i++)
		T(i) = dist(gen);

	return T;
}

CPUTensor CPUTensor::randomUniform(const vector<int>& shape, float start, float end)
{
	uniform_real_distribution<float> dist(start, end);

	CPUTensor T(shape);

	for (int i = 0; i < T.size(); i++)
		T(i) = dist(gen);

	return T;
}

void CPUTensor::recursMapping(vector<int>& I, const vector<int>& shape, const vector<int>& stride, int idx, int dim)
{
	int rank = shape.size();

	if (dim == rank)
	{
		I.push_back(idx);
		return;
	}

	for (int i = 0; i < shape[dim]; i++)
	{
		recursMapping(I, shape, stride, idx + i * stride[dim], dim + 1);
	}
}

/*CPUTensor CPUTensor::theMax(const CPUTensor& A, int axis)
{
	vector<int> newShape = A.shape;
	vector<int> newStride = A.stride;
	vector<int> I;

	newShape.erase(newShape.begin() + axis);
	newStride.erase(newStride.begin() + axis);

	CPUTensor C(newShape);

	recursMapping(I, newShape, newStride, 0, 0);

	//C = vector<float>(I.begin(), I.end());
	
	int stride = A.stride[axis];

	for (int i = 0; i < C.total; i++)
	{
		float the_max = -FLT_MAX;
		int partial_idx = I[i];

		for (int k = 0; k < A.shape[axis]; k++)
		{
			if (A.rawData()[partial_idx + k * stride] > the_max)
				the_max = A.rawData()[partial_idx + k * stride];
		}

		C.rawData()[i] = the_max;
	}

	return C;
}*/

CPUTensor CPUTensor::loadMatrixBin(const string& filepath, int rows, int cols)
{
	CPUTensor mat({ rows, cols });

	ifstream file(filepath, ios::binary);

	if (!file)
	{
		throw runtime_error("File " + filepath + " could not be opened");
	}

	file.read(
		reinterpret_cast<char*>(mat.data.f.data()),
		rows * cols * sizeof(float)
	);

	if (!file)
	{
		throw runtime_error("Error while reading file" + filepath);
	}

	return mat;
}

CPUTensor CPUTensor::loadTensorBin(
	const std::string& filepath,
	const std::vector<int>& shape)
{
	static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
		"The exporter requires IEEE-754 float32.");

	// The exported format is little-endian, as on your Windows x86-64 machine.
	const std::uint32_t one = 1;
	if (*reinterpret_cast<const unsigned char*>(&one) != 1)
		throw std::runtime_error("This loader requires a little-endian host.");

	if (shape.empty())
		throw std::runtime_error("Dataset shape cannot be empty!");

	// Your Tensor implementation uses int counts/indices.
	size_t count = 1;
	for (int dimension : shape)
	{
		if (dimension <= 0)
			throw std::runtime_error("Dataset dimensions must be positive!");

		if (count > static_cast<size_t>(std::numeric_limits<int>::max()) /
			static_cast<size_t>(dimension))
			throw std::runtime_error("Dataset exceeds int indexing capacity!");

		count *= static_cast<size_t>(dimension);
	}

	if (count > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) /
		sizeof(float))
		throw std::runtime_error("Dataset exceeds stream read capacity!");

	const std::streamsize bytes =
		static_cast<std::streamsize>(count * sizeof(float));

	std::ifstream file(filepath, std::ios::binary | std::ios::ate);
	if (!file)
		throw std::runtime_error("Could not open file: " + filepath);

	if (file.tellg() != std::streampos(bytes))
		throw std::runtime_error(
			"File size does not match the requested shape: " + filepath);

	file.seekg(0, std::ios::beg);

	// CPUTensor's default dtype is float32.
	CPUTensor mat(shape);
	file.read(reinterpret_cast<char*>(mat.data.f.data()), bytes);
	if (!file)
		throw std::runtime_error("Error reading file: " + filepath);

	return mat;
}
