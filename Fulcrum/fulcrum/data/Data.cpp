#include <fulcrum/data/Data.h>
#include <iostream>
#include <filesystem>
#include <numeric>
#include <random>
#include <fulcrum/io/IO.h>

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

    Dataset make_dataset(const cv::String& path, const vector<string>& classes, bool resize, int size)
    {
        filesystem::path root = path;
        int sample_counter = 0;
        vector<float> class_idxs;
        vector<float> data;
        vector<float> img_data;
        CPUTensor samples;
        CPUTensor labels;
        Dataset dataset(samples, labels);

        for (int i = 0; i < classes.size(); i++)
        {
            string class_name = classes[i];

            for (const auto& img_entry : filesystem::directory_iterator(root / class_name))
            {
                if(img_entry.is_regular_file())
                    std::cout << class_name << " -> " << img_entry.path().string() << '\n';

                if (!(img_entry.is_regular_file()))
                    continue;

                samples = io::load_img(img_entry.path().string(), resize, size);

                img_data = samples.getFloatVector();

                data.insert(data.end(), img_data.begin(), img_data.end());

                class_idxs.push_back(i);

                sample_counter++;
            }
        }

        if (samples.getShape().empty())
            throw runtime_error("No image was found!");

        dataset.samples.resize({ sample_counter, samples.getShape()[1], samples.getShape()[2], samples.getShape()[3] });
        dataset.samples = data;

        if (classes.size() == 2)
        {
            dataset.labels.resize({ 1, sample_counter });
            dataset.labels = class_idxs;
        }
        else
        {
            dataset.labels = CPUTensor::zeros({ static_cast<int>(classes.size()), sample_counter });
            for (int i = 0; i < class_idxs.size(); i++)
                dataset.labels(class_idxs[i] * sample_counter + i) = 1.0f;
        }
        dataset.class_labels = classes;

        return dataset;
    }

    DatasetSplits train_test_val_split(const Dataset& dataset, float test, float val, bool shuffle, int seed)
    {
        if (test < 0.0f)
            throw runtime_error("Test percentage value cannot be negative!");

        if (val < 0.0f)
            throw runtime_error("Validation percentage value cannot be negative!");

        if (dataset.samples.dim() != 4)
            throw runtime_error("Dataset samples must have shape (sample_count, C, H, W)!");

        if (dataset.labels.dim() != 2)
            throw runtime_error("Dataset labels must have shape (class_count, sample_count)!");

        int sample_count = dataset.samples.getShape()[0];
        vector<int> idxs(sample_count);
        iota(idxs.begin(), idxs.end(), 0);

        if (shuffle)
        {
            mt19937 rng(seed);
            std::shuffle(idxs.begin(), idxs.end(), rng);
        }

        int C = dataset.samples.getShape()[1];
        int H = dataset.samples.getShape()[2];
        int W = dataset.samples.getShape()[3];
        int classes = dataset.labels.getShape()[0];

        int test_samples = sample_count * test;
        int val_samples = (sample_count - test_samples) * val;
        int train_samples = sample_count - test_samples - val_samples;

        CPUTensor X_train({ train_samples, C, H, W });
        CPUTensor y_train({ classes, train_samples });

        CPUTensor X_test({ test_samples, C, H, W });
        CPUTensor y_test({ classes, test_samples });

        CPUTensor X_val({ val_samples, C, H, W });
        CPUTensor y_val({ classes, val_samples });

        Dataset train_data(X_train, y_train);
        Dataset test_data(X_test, y_test);
        Dataset val_data(X_val, y_val);

        DatasetSplits splits(train_data, test_data, val_data);

        if (train_samples >= val_samples && train_samples >= test_samples)
        {
            for (int i = 0; i < train_samples; i++)
            {
                int idx = idxs[i];
                for (int c = 0; c < C; c++)
                {
                    for (int h = 0; h < H; h++)
                    {
                        for (int w = 0; w < W; w++)
                        {
                            int train_idx = idx * C * H * W + c * H * W + h * W + w;
                            int train_splits_idx = i * C * H * W + c * H * W + h * W + w;
                            splits.train.samples(train_splits_idx) = dataset.samples(train_idx);
                            
                            if (i >= (train_samples - val_samples))
                            {
                                int new_idx = idxs[i + val_samples];
                                int new_splits_idx = i - (train_samples - val_samples);
                                int val_splits_idx = new_splits_idx * C * H * W + c * H * W + h * W + w;
                                int val_idx = new_idx * C * H * W + c * H * W + h * W + w;
                                splits.val.samples(val_splits_idx) = dataset.samples(val_idx);
                            }
                            if (i >= (train_samples - test_samples))
                            {
                                int new_idx = idxs[i + val_samples + test_samples];
                                int new_splits_idx = i - (train_samples - test_samples);
                                int test_splits_idx = new_splits_idx * C * H * W + c * H * W + h * W + w;
                                int test_idx = new_idx * C * H * W + c * H * W + h * W + w;
                                splits.test.samples(test_splits_idx) = dataset.samples(test_idx);
                            }             
                        }
                    }

                    for (int cl = 0; cl < classes; cl++)
                    {
                        int train_idx = cl * sample_count + idx;
                        int train_splits_idx = cl * train_samples + i;
                        splits.train.labels(train_splits_idx) = dataset.labels(train_idx);

                        if (i >= (train_samples - val_samples))
                        {
                            int new_idx = idxs[i + val_samples];
                            int new_splits_idx = i - (train_samples - val_samples);
                            int val_idx = cl * sample_count + new_idx;
                            int val_splits_idx = cl * val_samples + new_splits_idx;
                            splits.val.labels(val_splits_idx) = dataset.labels(val_idx);
                        }
                        if (i >= (train_samples - test_samples))
                        {
                            int new_idx = idxs[i + val_samples + test_samples];
                            int new_splits_idx = i - (train_samples - test_samples);
                            int test_idx = cl * sample_count + new_idx;
                            int test_splits_idx = cl * test_samples + new_splits_idx;
                            splits.test.labels(test_splits_idx) = dataset.labels(test_idx);
                        }
                    }
                }
            }
        }
        else if (val_samples >= train_samples && val_samples >= test_samples)
        {
            for (int i = 0; i < val_samples; i++)
            {
                int idx = idxs[i];
                for (int c = 0; c < C; c++)
                {
                    for (int h = 0; h < H; h++)
                    {
                        for (int w = 0; w < W; w++)
                        {
                            int val_idx = idx * C * H * W + c * H * W + h * W + w;
                            int val_splits_idx = i * C * H * W + c * H * W + h * W + w;
                            splits.val.samples(val_splits_idx) = dataset.samples(val_idx);

                            if (i >= (val_samples - train_samples))
                            {
                                int new_idx = idxs[i + train_samples];
                                int new_splits_idx = i - (val_samples - train_samples);
                                int train_idx = new_idx * C * H * W + c * H * W + h * W + w;
                                int train_splis_idx = new_splits_idx * C * H * W + c * H * W + h * W + w;
                                splits.train.samples(train_splis_idx) = dataset.samples(train_idx);
                            }
                            if (i >= (val_samples - test_samples))
                            {
                                int new_idx = idxs[i + train_samples + test_samples];
                                int new_splits_idx = i - (val_samples - test_samples);
                                int test_idx = new_idx * C * H * W + c * H * W + h * W + w;
                                int test_splits_idx = new_splits_idx * C * H * W + c * H * W + h * W + w;
                                splits.test.samples(test_splits_idx) = dataset.samples(test_idx);
                            }
                        }
                    }
                }

                for (int cl = 0; cl < classes; cl++)
                {
                    int val_idx = cl * sample_count + idx;
                    int val_splits_idx = cl * val_samples + i;
                    splits.val.labels(val_splits_idx) = dataset.labels(val_idx);

                    if (i >= (val_samples - train_samples))
                    {
                        int new_idx = idxs[i + train_samples];
                        int new_splits_idx = i - (val_samples - train_samples);
                        int train_idx = cl * sample_count + new_idx;
                        int train_splits_idx = cl * train_samples + new_splits_idx;
                        splits.train.labels(train_splits_idx) = dataset.labels(train_idx);
                    }
                    if (i >= (val_samples - test_samples))
                    {
                        int new_idx = idxs[i + train_samples + test_samples];
                        int new_splits_idx = i - (val_samples - test_samples);
                        int test_idx = cl * sample_count + new_idx;
                        int test_splits_idx = cl * test_samples + new_splits_idx;
                        splits.test.labels(test_splits_idx) = dataset.labels(test_idx);
                    }
                }
            }
        }
        else
        {
            for (int i = 0; i < test_samples; i++)
            {
                int idx = idxs[i];
                for (int c = 0; c < C; c++)
                {
                    for (int h = 0; h < H; h++)
                    {
                        for (int w = 0; w < W; w++)
                        {
                            int test_idx = idx * C * H * W + c * H * W + h * W + w;
                            int test_splits_idx = i * C * H * W + c * H * W + h * W + w;
                            splits.test.samples(test_splits_idx) = dataset.samples(test_idx);

                            if (i >= (test_samples - train_samples))
                            {
                                int new_idx = idxs[i + train_samples];
                                int new_splits_idx = i - (test_samples - train_samples);
                                int train_idx = new_idx * C * H * W + c * H * W + h * W + w;
                                int train_splits_idx = new_splits_idx * C * H * W + c * H * W + h * W + w;
                                splits.train.samples(train_splits_idx) = dataset.samples(train_idx);
                            }

                            if (i >= (test_samples - val_samples))
                            {
                                int new_idx = idxs[i + train_samples + val_samples];
                                int new_splits_idx = i - (test_samples - val_samples);
                                int val_idx = new_idx * C * H * W + c * H * W + h * W + w;
                                int val_splits_idx = new_splits_idx * C * H * W + c * H * W + h * W + w;
                                splits.val.samples(val_splits_idx) = dataset.samples(val_idx);
                            }
                        }
                    }
                }

                for (int cl = 0; cl < classes; cl++)
                {
                    int test_idx = cl * sample_count + idx;
                    int test_splits_idx = cl * test_samples + i;
                    splits.test.labels(test_splits_idx) = dataset.labels(test_idx);

                    if (i >= (test_samples - train_samples))
                    {
                        int new_idx = idxs[i + train_samples];
                        int new_splits_idx = i - (test_samples - train_samples);
                        int train_idx = cl * sample_count + new_idx;
                        int train_splits_idx = cl * train_samples + new_splits_idx;
                        splits.train.labels(train_splits_idx) = dataset.labels(train_idx);
                    }

                    if (i >= (test_samples - val_samples))
                    {
                        int new_idx = idxs[i + train_samples + val_samples];
                        int new_splits_idx = i - (test_samples - val_samples);
                        int val_idx = cl * sample_count + new_idx;
                        int val_splits_idx = cl * val_samples + new_splits_idx;
                        splits.val.labels(val_splits_idx) = dataset.labels(val_idx);
                    }
                }
            }
        }

        return splits;
    }
}