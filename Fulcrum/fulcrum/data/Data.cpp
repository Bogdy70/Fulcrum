#include <fulcrum/data/Data.h>
#include <iostream>
#include <filesystem>

using namespace std;

namespace fulcrum::data
{
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

    CPUTensor load_img(const cv::String& path)
    {
        std::cout << "File exists: "
            << std::filesystem::exists(path) << '\n';

        cv::Mat loaded = cv::imread(path, cv::IMREAD_COLOR);

        if (loaded.empty())
            throw runtime_error("Unable to read image at " + path);

        if (loaded.rows != 1080 || loaded.cols != 1920)
            throw runtime_error("Incorrect image resolution!");

        cv::Mat resized;
        cv::resize(loaded, resized, cv::Size(128, 72), 0.0, 0.0, cv::INTER_LANCZOS4); // cv::Size(width, height)

        cv::Mat padded;
        cv::copyMakeBorder(resized, padded, 28, 28, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        CPUTensor loaded_img({ 1, 3, 128, 128 });

        for (int i = 0; i < padded.rows; i++)
        {
            for (int j = 0; j < padded.cols; j++)
            {
                cv::Vec3b pixel = padded.at<cv::Vec3b>(i, j);

                loaded_img(i * 128 + j) = pixel[2] / 255.0f;
                loaded_img(128 * 128 + i * 128 + j) = pixel[1] / 255.0f;
                loaded_img(2 * 128 * 128 + i * 128 + j) = pixel[0] / 255.0f;
            }
        }

        return loaded_img;
    }
}