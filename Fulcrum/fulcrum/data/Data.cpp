#include <fulcrum/data/Data.h>
#include <iostream>
#include <filesystem>

using namespace std;

namespace fulcrum::data
{
    int DataLoader::size() const
    {
        return ceil(static_cast<float>(samples.getShape()[0]) / batch_size);
    }

    int DataLoader::nr_samples() const
    {
        return samples.getShape()[0];
    }

    int DataLoader::nr_labels() const
    {
        return labels.getShape()[1];
    }

    DataLoader::Batch DataLoader::operator[](int index)
    {
        DataLoader::Batch batch;

        int C = samples.getShape()[1];
        int H = samples.getShape()[2];
        int W = samples.getShape()[3];
        int classes = labels.getShape()[0];

        int save_batch_size = index == size() - 1 && static_cast<float>(samples.getShape()[0]) / batch_size - index != 0 ? samples.getShape()[0] - (index * batch_size) : batch_size;

        batch.sample = CPUTensor({ save_batch_size, C, H, W });
        batch.label = CPUTensor({ classes, save_batch_size });

        for (int b = batch_size * index; b < (batch_size * index) + save_batch_size; b++)
        {
            for (int c = 0; c < C; c++)
            {
                for (int h = 0; h < H; h++)
                {
                    for (int w = 0; w < W; w++)
                    {
                        int other_idx = b * C * H * W + c * H * W + h * W + w;
                        int save_idx = (b - index * batch_size) * C * H * W + c * H * W + h * W + w;

                        batch.sample(save_idx) = samples(other_idx);
                    }
                }
            }
        }

        for (int cl = 0; cl < classes; cl++)
        {
            for (int b = batch_size * index; b < (batch_size * index) + save_batch_size; b++)
            {
                int other_idx = cl * nr_labels() + b;
                int save_idx = cl * save_batch_size + (b - index * batch_size);

                batch.label(save_idx) = labels(other_idx);
            }
        }

        return batch;
    }

    const DataLoader::Batch DataLoader::operator[](int index) const
    {
        DataLoader::Batch batch;

        int C = samples.getShape()[1];
        int H = samples.getShape()[2];
        int W = samples.getShape()[3];
        int classes = labels.getShape()[0];

        int save_batch_size = index == size() - 1 && static_cast<float>(samples.getShape()[0]) / batch_size - index != 0 ? samples.getShape()[0] - (index * batch_size) : batch_size;

        batch.sample = CPUTensor({ save_batch_size, C, H, W });
        batch.label = CPUTensor({ classes, save_batch_size });

        for (int b = batch_size * index; b < (batch_size * index) + save_batch_size; b++)
        {
            for (int c = 0; c < C; c++)
            {
                for (int h = 0; h < H; h++)
                {
                    for (int w = 0; w < W; w++)
                    {
                        int other_idx = b * C * H * W + c * H * W + h * W + w;
                        int save_idx = (b - index * batch_size) * C * H * W + c * H * W + h * W + w;

                        batch.sample(save_idx) = samples(other_idx);
                    }
                }
            }
        }

        for (int cl = 0; cl < classes; cl++)
        {
            for (int b = batch_size * index; b < (batch_size * index) + save_batch_size; b++)
            {
                int other_idx = cl * nr_labels() + b;
                int save_idx = cl * save_batch_size + (b - index * batch_size);

                batch.label(save_idx) = labels(other_idx);
            }
        }

        return batch;
    }

    void printMnistImage(const CPUTensor& img, int sample_idx)
    {
        for (int i = 0; i < 784; i++)
        {
            if (i % 28 == 0)
                cout << "\n";

            float value = img(i * img.getShape()[img.dim() - 1] + sample_idx);

            if (value > 0.7f)
                cout << "#";
            else if (value > 0.3f)
                cout << ".";
            else
                cout << " ";
        }
    }

    float choose_label(const Tensor& y, int idx)
    {
        if (y.dim() != 2)
            throw runtime_error("Tensor must be 2 dimensional!");

        if (idx >= y.getShape()[1])
            throw runtime_error("Invalid label index!");

        if (y.getShape()[0] == 1)
            return y.toCPU()(idx);

        Tensor y_labels = Tensor::argmax(y, 0);

        return y_labels.toCPU()(idx);
    }

    CPUTensor choose_img(const CPUTensor& X, int idx)
    {
        if (X.dim() != 4)
            throw runtime_error("Tensor must be 4 dimensional!");

        if (idx >= X.getShape()[0])
            throw runtime_error("Invalid image index!");

        int actual_idx = idx * X.getStride()[0];

        vector<int> new_shape;
        new_shape.push_back(1);

        for (int i = 1; i < X.getShape().size(); i++)
        {
            new_shape.push_back(X.getShape()[i]);
        }

        CPUTensor one_img(new_shape);

        for (int i = 0; i < X.getShape()[1]; i++)
        {
            for (int j = 0; j < X.getShape()[2]; j++)
            {
                for (int k = 0; k < X.getShape()[3]; k++)
                {
                    int one_idx = i * X.getStride()[1] + j * X.getStride()[2] + k * X.getStride()[3];
                    int the_idx = actual_idx + i * X.getStride()[1] + j * X.getStride()[2] + k * X.getStride()[3];

                    one_img(one_idx) = X(the_idx);
                }
            }
        }

        return one_img;
    }

    CPUTensor load_img(const cv::String& path, int size)
    {
        if (size <= 0)
            throw runtime_error("Size cannot be negative!");

        std::cout << "File exists: "
            << std::filesystem::exists(path) << '\n';

        cv::Mat loaded = cv::imread(path, cv::IMREAD_COLOR);

        if (loaded.empty())
            throw runtime_error("Unable to read image at " + path);

        if (loaded.rows < size || loaded.cols < size)
            throw runtime_error("Image size cannot be smaller than specified size!");

        cv::Mat padded;
        cv::Mat resized;
        int height, width, gap;

        if (loaded.cols > loaded.rows)
        {
            height = lround(static_cast<double>(loaded.rows) / loaded.cols * size);
            cv::resize(loaded, resized, cv::Size(size, height), 0.0, 0.0, cv::INTER_LANCZOS4);

            gap = (size - height);
            cv::copyMakeBorder(resized, padded, gap-(gap/2), gap/2, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        }
        else if (loaded.rows > loaded.cols)
        {
            width = lround(static_cast<double>(loaded.cols) / loaded.rows * size);
            cv::resize(loaded, resized, cv::Size(width, size), 0.0, 0.0, cv::INTER_LANCZOS4);

            gap = (size - width);
            cv::copyMakeBorder(resized, padded, 0, 0, gap - (gap / 2), gap / 2, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        }
        else
        {
            cv::resize(loaded, padded, cv::Size(size, size), 0.0, 0.0, cv::INTER_LANCZOS4);
        }

        CPUTensor loaded_img({ 1, 3, size, size });

        for (int i = 0; i < padded.rows; i++)
        {
            for (int j = 0; j < padded.cols; j++)
            {
                cv::Vec3b pixel = padded.at<cv::Vec3b>(i, j);

                loaded_img(i * size + j) = pixel[2] / 255.0f;
                loaded_img(size * size + i * size + j) = pixel[1] / 255.0f;
                loaded_img(2 * size * size + i * size + j) = pixel[0] / 255.0f;
            }
        }

        return loaded_img;
    }

    CPUTensor loadMatrixBin(const string& filepath, int rows, int cols)
    {
        CPUTensor mat({ rows, cols });

        ifstream file(filepath, ios::binary);

        if (!file)
        {
            throw runtime_error("File " + filepath + " could not be opened");
        }

        file.read(
            reinterpret_cast<char*>(mat.getFloatData()),
            rows * cols * sizeof(float)
        );

        if (!file)
        {
            throw runtime_error("Error while reading file" + filepath);
        }

        return mat;
    }

    CPUTensor loadTensorBin(
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
        file.read(reinterpret_cast<char*>(mat.getFloatData()), bytes);
        if (!file)
            throw std::runtime_error("Error reading file: " + filepath);

        return mat;
    }
}