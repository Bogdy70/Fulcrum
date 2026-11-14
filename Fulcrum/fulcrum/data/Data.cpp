#include <fulcrum/data/Data.h>
#include <iostream>

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
}