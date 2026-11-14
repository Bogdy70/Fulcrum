#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <chrono>
#include <stdexcept>
#include <fstream>
#include <filesystem>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <Fulcrum.h>

using namespace std;
using fulcrum::Tensor;
using fulcrum::MaxPoolRes;
using fulcrum::CPUTensor;
using fulcrum::DataType;
using namespace fulcrum::io;
using namespace fulcrum::nn;
using namespace fulcrum::optim;
using namespace fulcrum::data;

bool same_tensor(const Tensor& A, const Tensor& B)
{
    if (A.getDataType() != B.getDataType())
        return false;

    if (A.getShape() != B.getShape())
        return false;

    if (A.size() == 0)
        return true;

    CPUTensor cpuA = A.toCPU();
    CPUTensor cpuB = B.toCPU();

    std::size_t count = static_cast<std::size_t>(A.size());

    switch (A.getDataType())
    {
    case DataType::float32:
        return std::memcmp(
            cpuA.getFloatData(),
            cpuB.getFloatData(),
            count * sizeof(float)) == 0;

    case DataType::int32:
        return std::memcmp(
            cpuA.getIntData(),
            cpuB.getIntData(),
            count * sizeof(int32_t)) == 0;

    default:
        return false;
    }
}

bool compare_params(const Parameters& original,
    const Parameters& loaded)
{
    if (original.W.size() != loaded.W.size() ||
        original.B.size() != loaded.B.size() ||
        original.W.size() != original.B.size())
    {
        std::cout << "FAIL: parameter vector sizes do not match!\n";
        return false;
    }

    bool all_match = true;

    // Index 0 is unused in your current Parameters convention.
    for (std::size_t l = 1; l < original.W.size(); l++)
    {
        bool weights_match = same_tensor(original.W[l], loaded.W[l]);
        bool biases_match = same_tensor(original.B[l], loaded.B[l]);

        std::cout << "W[" << l << "]: "
            << (weights_match ? "PASS" : "FAIL") << '\n';

        std::cout << "B[" << l << "]: "
            << (biases_match ? "PASS" : "FAIL") << '\n';

        all_match = all_match && weights_match && biases_match;
    }

    std::cout << "\nOverall: "
        << (all_match ? "PASS" : "FAIL") << '\n';

    return all_match;
}

bool compare_cnn(const CNN_model& original,
    const CNN_model& loaded)
{
    cout << "\nParameter comparison\n\n";

    bool all_match = compare_params(
        original.params,
        loaded.params);

    auto check = [&](const string& name, bool matches)
        {
            cout << name << ": "
                << (matches ? "PASS" : "FAIL") << '\n';

            all_match = all_match && matches;
        };

    cout << "\nModel configuration comparison\n\n";

    check(
        "Convolution configuration count",
        original.cv_stats.size() == loaded.cv_stats.size());

    for (size_t i = 0;
        i < original.cv_stats.size() &&
        i < loaded.cv_stats.size();
        i++)
    {
        const auto& a = original.cv_stats[i];
        const auto& b = loaded.cv_stats[i];

        bool matches =
            a.in_channels == b.in_channels &&
            a.out_channels == b.out_channels &&
            a.kernel_size == b.kernel_size &&
            a.hStride == b.hStride &&
            a.vStride == b.vStride &&
            a.padding == b.padding;

        check("Conv[" + to_string(i) + "]", matches);
    }

    check(
        "Max-pool configuration count",
        original.mx_stats.size() == loaded.mx_stats.size());

    for (size_t i = 0;
        i < original.mx_stats.size() &&
        i < loaded.mx_stats.size();
        i++)
    {
        const auto& a = original.mx_stats[i];
        const auto& b = loaded.mx_stats[i];

        bool matches =
            a.kernel_size == b.kernel_size &&
            a.hStride == b.hStride &&
            a.vStride == b.vStride &&
            a.padding == b.padding;

        check("MaxPool[" + to_string(i) + "]", matches);
    }

    check("Activation", original.activ == loaded.activ);

    cout << "Original activation: " << original.activ << '\n';
    cout << "Loaded activation:   " << loaded.activ << '\n';

    cout << "\nOverall CNN comparison: "
        << (all_match ? "PASS" : "FAIL") << '\n';

    return all_match;
}

void numerical_gradient_test(
    Parameters& params,
    const Tensor& X,
    const Tensor& Y,
    const vector<convStats>& cv_stats,
    const vector<MaxPoolStats>& mx_stats,
    const string& activation,
    float epsilon = 1e-3f,
    double abs_tolerance = 2e-3,
    double rel_tolerance = 2e-2)
{
    if (epsilon <= 0.0f)
        throw runtime_error("Epsilon must be positive!");

    auto checkCuda = [](cudaError_t error)
        {
            if (error != cudaSuccess)
                throw runtime_error(cudaGetErrorString(error));
        };

    auto readTensor = [&](const Tensor& tensor)
        {
            size_t count = 1;

            for (int dimension : tensor.getShape())
                count *= static_cast<size_t>(dimension);

            vector<float> values(count);

            checkCuda(cudaMemcpy(
                values.data(),
                tensor.getFloatData(),
                count * sizeof(float),
                cudaMemcpyDeviceToHost
            ));

            return values;
        };

    auto writeElement = [&](Tensor& tensor, size_t index, float value)
        {
            checkCuda(cudaMemcpy(
                tensor.getFloatData() + index,
                &value,
                sizeof(float),
                cudaMemcpyHostToDevice
            ));
        };

    int L = static_cast<int>(params.W.size());

    // Dropout and regularization are disabled throughout this test.
    auto calculateLoss = [&]() -> double
        {
            Forward cache = conv_frdpass(
                params, X, cv_stats, mx_stats, activation, 0.0f
            );

            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());

            return static_cast<double>(
                cost(Y, cache.A[L - 1], params, 0.0f, 0.0f)
                );
        };

    // Analytical gradients at the original parameter values.
    Forward cache = conv_frdpass(
        params, X, cv_stats, mx_stats, activation, 0.0f
    );

    Backward grads = conv_bckprop(
        cache, params, Y,
        cv_stats, mx_stats, activation, 0.0f
    );

    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    size_t totalChecked = 0;
    size_t totalFailed = 0;

    auto oldPrecision = cout.precision();
    cout << setprecision(8);
    cout << "\nNumerical gradient test\n";

    for (int l = 1; l < L; l++)
    {
        for (bool isBias : { false, true })
        {
            Tensor& parameter =
                isBias ? params.B[l] : params.W[l];

            const Tensor& gradient =
                isBias ? grads.dB[l] : grads.dW[l];

            string name =
                string(isBias ? "B[" : "W[") + to_string(l) + "]";

            if (parameter.getShape() != gradient.getShape())
                throw runtime_error(name + ": gradient shape mismatch!");

            vector<float> original = readTensor(parameter);
            vector<float> analytical = readTensor(gradient);

            size_t failed = 0;
            double largestError = 0.0;

            cout << "\n" << name << "\n";

            for (size_t i = 0; i < original.size(); i++)
            {
                // Use the actual representable float perturbations.
                float plusValue = original[i] + epsilon;
                float minusValue = original[i] - epsilon;

                if (plusValue == minusValue)
                    throw runtime_error("Epsilon too small for parameter!");

                double plusLoss;
                double minusLoss;

                try
                {
                    writeElement(parameter, i, plusValue);
                    plusLoss = calculateLoss();

                    writeElement(parameter, i, minusValue);
                    minusLoss = calculateLoss();
                }
                catch (...)
                {
                    writeElement(parameter, i, original[i]);
                    throw;
                }

                // Restore this parameter before checking the next one.
                writeElement(parameter, i, original[i]);

                double numerical =
                    (plusLoss - minusLoss) /
                    (static_cast<double>(plusValue) - minusValue);

                double backward = analytical[i];
                double error = std::abs(numerical - backward);

                double allowedError =
                    abs_tolerance +
                    rel_tolerance *
                    std::max(std::abs(numerical), std::abs(backward));

                bool passed =
                    std::isfinite(numerical) &&
                    std::isfinite(backward) &&
                    error <= allowedError;

                if (!passed)
                    failed++;

                if (std::isfinite(error))
                    largestError = std::max(largestError, error);

                // Print a few examples, plus every failure.
                if (i < 3 || !passed)
                {
                    cout << "  Element " << i
                        << " | backward: " << backward
                        << " | numerical: " << numerical
                        << " | error: " << error
                        << " | " << (passed ? "PASS" : "FAIL")
                        << "\n";
                }
            }

            totalChecked += original.size();
            totalFailed += failed;

            cout << "  Checked: " << original.size()
                << " | Failed: " << failed
                << " | Largest finite absolute error: "
                << largestError << "\n";
        }
    }

    cout << "\nOverall: "
        << (totalFailed == 0 ? "PASS" : "FAIL")
        << " | Checked: " << totalChecked
        << " | Failed: " << totalFailed << "\n";

    cout.precision(oldPrecision);
}

int main()
{
    try
    {
        CPUTensor X_train_cat = CPUTensor::loadMatrixBin("data/cat/X_train.bin", 12288, 209);
        CPUTensor y_train_cat = CPUTensor::loadMatrixBin("data/cat/Y_train.bin", 1, 209);

        CPUTensor X_test_cat = CPUTensor::loadMatrixBin("data/cat/X_test.bin", 12288, 50);
        CPUTensor y_test_cat = CPUTensor::loadMatrixBin("data/cat/Y_test.bin", 1, 50);

        CPUTensor X_train_mnist = CPUTensor::loadMatrixBin("data/mnist/X_train.bin", 784, 5000);
        CPUTensor y_train_mnist = CPUTensor::loadMatrixBin("data/mnist/Y_train.bin", 10, 5000);

        CPUTensor X_test_mnist = CPUTensor::loadMatrixBin("data/mnist/X_test.bin", 784, 1000);
        CPUTensor y_test_mnist = CPUTensor::loadMatrixBin("data/mnist/Y_test.bin", 10, 1000);

        CPUTensor X_train_cnn = CPUTensor::loadTensorBin("data/cifar10_cnn/X_train.bin", { 1000, 3, 32, 32 });
        CPUTensor y_train_cnn = CPUTensor::loadTensorBin("data/cifar10_cnn/Y_train.bin", { 10, 1000 });

        CPUTensor X_test_cnn = CPUTensor::loadTensorBin("data/cifar10_cnn/X_test.bin", { 100, 3, 32, 32 });
        CPUTensor y_test_cnn = CPUTensor::loadTensorBin("data/cifar10_cnn/Y_test.bin", { 10, 100 });

        CPUTensor X_train_meteors = CPUTensor::loadTensorBin("data/meteor_cnn/X_train.bin", { 1000, 3, 128, 128 });
        CPUTensor y_train_meteors = CPUTensor::loadTensorBin("data/meteor_cnn/Y_train.bin", { 1, 1000 });

        CPUTensor X_val_meteors = CPUTensor::loadTensorBin("data/meteor_cnn/X_val.bin", { 100, 3, 128, 128 });
        CPUTensor y_val_meteors = CPUTensor::loadTensorBin("data/meteor_cnn/Y_val.bin", { 1, 100 });

        CPUTensor X_test_meteors = CPUTensor::loadTensorBin("data/meteor_cnn/X_test.bin", { 100, 3, 128, 128 });
        CPUTensor y_test_meteors = CPUTensor::loadTensorBin("data/meteor_cnn/Y_test.bin", { 1, 100 });

        cout << "Cat dataset loaded successfully\n";

        cout << "X_train_cat: (" << X_train_cat.getShape()[X_train_cat.dim() - 2] << ", " << X_train_cat.getShape()[X_train_cat.dim() - 1] << ")\n";
        cout << "y_train_cat: (" << y_train_cat.getShape()[y_train_cat.dim() - 2] << ", " << y_train_cat.getShape()[y_train_cat.dim() - 1] << ")\n";
        
        cout << "X_test_cat: (" << X_test_cat.getShape()[X_test_cat.dim() - 2] << ", " << X_test_cat.getShape()[X_test_cat.dim() - 1] << ")\n";
        cout << "y_test_cat: (" << y_test_cat.getShape()[y_test_cat.dim() - 2] << ", " << y_test_cat.getShape()[y_test_cat.dim() - 1] << ")\n";

        cout << "\nMnist dataset loaded successfully\n";

        cout << "X_train_mnist: (" << X_train_mnist.getShape()[X_train_mnist.dim() - 2] << ", " << X_train_mnist.getShape()[X_train_mnist.dim() - 1] << ")\n";
        cout << "y_train_mnist: (" << y_train_mnist.getShape()[y_train_mnist.dim() - 2] << ", " << y_train_mnist.getShape()[y_train_mnist.dim() - 1] << ")\n";

        cout << "X_test_mnist: (" << X_test_mnist.getShape()[X_test_mnist.dim() - 2] << ", " << X_test_mnist.getShape()[X_test_mnist.dim() - 1] << ")\n";
        cout << "y_test_mnist: (" << y_test_mnist.getShape()[y_test_mnist.dim() - 2] << ", " << y_test_mnist.getShape()[y_test_mnist.dim() - 1] << ")\n";

        cout << "\nCifar10 dataset loaded successfully";

        cout << "\nX_train_cnn: (";
        for (int i = 0; i < X_train_cnn.dim(); i++)
        {
            cout << X_train_cnn.getShape()[i];
            if (i != X_train_cnn.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\ny_train_cnn: (";
        for (int i = 0; i < y_train_cnn.dim(); i++)
        {
            cout << y_train_cnn.getShape()[i];
            if (i != y_train_cnn.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\nX_test_cnn: (";
        for (int i = 0; i < X_test_cnn.dim(); i++)
        {
            cout << X_test_cnn.getShape()[i];
            if (i != X_test_cnn.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\ny_test_cnn: (";
        for (int i = 0; i < y_test_cnn.dim(); i++)
        {
            cout << y_test_cnn.getShape()[i];
            if (i != y_test_cnn.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\n\nMeteors dataset loaded successfully";

        cout << "\nX_train_meteors: (";
        for (int i = 0; i < X_train_meteors.dim(); i++)
        {
            cout << X_train_meteors.getShape()[i];
            if (i != X_train_meteors.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\ny_train_meteors: (";
        for (int i = 0; i < y_train_meteors.dim(); i++)
        {
            cout << y_train_meteors.getShape()[i];
            if (i != y_train_meteors.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\nX_validation_meteors: (";
        for (int i = 0; i < X_val_meteors.dim(); i++)
        {
            cout << X_val_meteors.getShape()[i];
            if (i != X_val_meteors.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\ny_validation_meteors: (";
        for (int i = 0; i < y_val_meteors.dim(); i++)
        {
            cout << y_val_meteors.getShape()[i];
            if (i != y_val_meteors.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\nX_test_meteors: (";
        for (int i = 0; i < X_test_meteors.dim(); i++)
        {
            cout << X_test_meteors.getShape()[i];
            if (i != X_test_meteors.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\ny_test_meteors: (";
        for (int i = 0; i < y_test_meteors.dim(); i++)
        {
            cout << y_test_meteors.getShape()[i];
            if (i != y_test_meteors.dim() - 1)
                cout << ", ";
        }
        cout << ")";


        vector<int> dim_list = { X_train_cat.getShape()[X_train_cat.dim() - 2], 100, 100, 200, y_train_cat.getShape()[y_train_cat.dim() - 2] };
        auto start = std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed;

        cout << "\n\nTensor test\n\n";

        Tensor T({ 2, 3, 4, 2 });
        T = vector<float> { 1, 2, 3, 4, 5, 6, 7, 8, 9, 127, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48 };
        CPUTensor CPUT = T.toCPU();
        CPUT.print();

        cout << "\n\nZeros test\n\n";

        Tensor Z = Tensor::zeros({ 2, 3, 3, 3 });
        Z.toCPU().print();


        cout << "\n\nRandom test\n\n";

        CPUTensor R = CPUTensor::random({ 2, 3, 3 });
        R.print();

        cout << "\n";

        CPUTensor RU = CPUTensor::randomUniform({ 1, 3, 3 }, 0.0f, 2.0f);
        RU.print();

        cout << "\n";

        CPUTensor::setSeed(42);
        Tensor RC = Tensor::random({ 1, 3, 3 });
        RC.toCPU().print();

        cout << "\n";
        
        CPUTensor::setSeed(42);
        Tensor RC1 = Tensor::random({ 2, 2, 3 });
        RC1.toCPU().print();

        cout << "\n\nFill test\n\n";

        Tensor F = Tensor::fill({ 3, 3, 2 }, 3.7f);
        F.toCPU().print();

        cout << "\n\nReshape test\n\n";

        Tensor S({ 2, 3, 3 });
        S = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 };
        S.toCPU().print();

        cout << "\n";

        S.reshape({ 3, 6 });
        S.toCPU().print();

        cout << "\n\nResize test\n\n";

        Tensor Rs({ 2, 3, 3 });
        Rs = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 };
        Rs.toCPU().print();

        cout << "\n";

        Rs.resize({ 2, 3, 2, 2 });
        Rs.toCPU().print();

        cout << "\n\nMul test\n\n";

        Tensor A({ 2, 3, 3 });
        A = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 };
        Tensor B({ 2, 3, 3 });
        B = vector<float>{ 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

        Tensor C = B * A;
        C.toCPU().print();

        cout << "\n\nDiv test\n\n";

        Tensor C1 = B / B;
        C1.toCPU().print();

        cout << "\n\nAdd test\n\n";

        Tensor C2 = B + A;
        C2.toCPU().print();

        cout << "\n\nSub test\n\n";

        Tensor C3 = B - A;
        C3.toCPU().print();

        cout << "\n\nScalar mul test\n\n";

        Tensor C4 = 2.0f * A;
        C4.toCPU().print();

        cout << "\n\nScalar div test\n\n";

        Tensor C5 = 2.0f / B;
        C5.toCPU().print();

        cout << "\n\nScalar add test\n\n";

        Tensor C6 = 7.0f + B;
        C6.toCPU().print();

        cout << "\n\nScalar sub test\n\n";

        Tensor C7 = 8.0f - A;
        C7.toCPU().print();
        
        cout << "\n\nSqueeze test\n\n";

        Tensor C8({ 1, 2, 3, 1 });
        C8.squeeze();
        C8.toCPU().print();
        cout << "\n(";
        for (int i = 0; i < C8.toCPU().dim(); i++)
        {
            if (i != C8.toCPU().dim() - 1)
                cout << C8.toCPU().getShape()[i] << ", ";
            else
                cout << C8.toCPU().getShape()[i];
        }
        cout << ")";

        cout << "\n\nUnsqueeze test\n\n";

        Tensor C9({4});
        C9 = vector<float>{ 1, 2, 3, 4 };
        C9.unsqueeze(1);
        C9.toCPU().print();
        cout << "\n(";
        for (int i = 0; i < C9.toCPU().dim(); i++)
        {
            if (i != C9.toCPU().dim() - 1)
                cout << C9.toCPU().getShape()[i] << ", ";
            else
                cout << C9.toCPU().getShape()[i];
        }
        cout << ")";

        cout << "\n\nPowT test\n\n";

        Tensor C10 = Tensor::powT(A, 2.0f);
        C10.toCPU().print();

        cout << "\n\nSqrtT test\n\n";

        Tensor C11 = Tensor::sqrtT(A);
        C11.toCPU().print();

        cout << "\n\nClipT test\n\n";

        Tensor C12 = Tensor::clipT(A, 3.0f, 16.0f);
        C12.toCPU().print();

        cout << "\n\nToScalar test\n\n";

        Tensor C13({ 1, 1 });
        C13 = vector<float>{ 12 };
        C13.toCPU().print();
        cout << "\n" << C13.toScalar();

        cout << "\n\nEquals test\n\n";

        Tensor D({ 2, 3, 3 });
        D = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 11, 12, 13, 27, 15, 52, 17, 18 };
        Tensor C14 = A == D;
        C14.toCPU().print();

        cout << "\n\nGreater than test\n\n";

        Tensor C15 = A > 7.0f;
        C15.toCPU().print();

        cout << "\n\nLess than test\n\n";

        Tensor C16 = B < 7.0f;
        C16.toCPU().print();

        cout << "\n\nMatmul test\n\n";

        Tensor A1({ 2, 3 });
        A1 = vector<float>{ 1, 2, 3, 4, 5, 6 };
        cout << "\n";
        A1.toCPU().print();

        Tensor B1({ 3, 4 });
        B1 = vector<float>{ 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 };
        cout << "\n";
        B1.toCPU().print();

        cout << "\n\nTranspose test\n\n";

        Tensor C18 = A1.T();
        C18.toCPU().print();

        cout << "\n\nBroadcast addition test\n\n";

        Tensor E({ 3, 1 });
        E = vector<float>{ 1, 2, 3 };

        Tensor C19 = B1 + E;
        C19.toCPU().print();

        cout << "\n\nBroadcast division test\n\n";

        Tensor F1({ 1, 4 });
        F1 = vector<float>{ 1, 2, 3, 4 };

        Tensor C20 = B1 / F1;
        C20.toCPU().print();

        cout << "\n\nSum test\n\n";

        Tensor C21 = Tensor::sum(A1);
        float x = C21.toScalar();
        cout << x << "\n\n";

        Tensor C22 = Tensor::sum(A1, 0);
        C22.toCPU().print();
        cout << "\n";

        Tensor C23 = Tensor::sum(A1, 1);
        C23.toCPU().print();

        cout << "\n\nArgmax test\n\n";

        Tensor C24 = Tensor::argmax(B1);
        C24.toCPU().print();
        cout << "\n";

        Tensor C25 = Tensor::argmax(B1, 1);
        C25.toCPU().print();

        cout << "\n\nMax test\n\n";

        cout << Tensor::maxT(B1).toScalar() << "\n\n";

        Tensor C26 = Tensor::maxT(B1, 0);
        C26.toCPU().print();
        cout << "\n";

        Tensor C27 = Tensor::maxT(B1, 1);
        C27.toCPU().print();
        cout << "\n";

        cout << "\n\nNew max test\n\n";

        Tensor G = Tensor::maxT(T, 3, true);
        G.toCPU().print();

        cout << "\n\nNew argmax test\n\n";

        Tensor Arg = Tensor::argmax(T);
        Arg.toCPU().print();

        cout << "\n\nNew sum test\n\n";

        Tensor Sum = Tensor::sum(T, 3, true);
        Sum.toCPU().print();

        cout << "\n\nMatmul test\n\n";

        Tensor T2({ 2, 3, 2, 5 });
        T2 = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 127, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59 , 60 };

        Tensor Matmul = T.matmul(T2);
        Matmul.toCPU().print();
        cout << "\n(";
        for (int i = 0; i < Matmul.dim(); i++)
        {
            cout << Matmul.getShape()[i];
            if (i != Matmul.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\n\nT test\n\n";

        Tensor Tr = T2.T();
        Tr.toCPU().print();
        cout << "\n(";
        for (int i = 0; i < Tr.dim(); i++)
        {
            cout << Tr.getShape()[i];
            if (i != Tr.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\n\nBroadcasting test\n\n";

        Tensor A2({ 3, 4, 2 });
        Tensor B2({ 3, 1, 2 });
        Tensor A3({ 2 });
        Tensor B3({ 3, 4, 2 });
        Tensor T3({ 2, 1, 4, 1, 2 });
        Tensor T4({ 1, 3, 1, 5, 2 });
        Tensor T5({ 1 });
        Tensor T6({ 1, 3, 1, 1 });
        Tensor T7({ 2, 1, 1, 5 });
        Tensor T8({ 2, 3, 4, 5 });
        Tensor T9({ 2, 3, 4, 5, 2 });
        Tensor T10({ 2, 1, 4, 1 });
        Tensor T11({ 1, 3, 1, 5 });
        Tensor T12({ 2, 3 });
        Tensor T13({ 2, 4 });

        T7 = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        T8 = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120 };
        T9 = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240 };
        T10 = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8 };
        T11 = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
        T12 = vector<float>{ 1, 2, 3, 4, 5, 6 };
        T13 = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8 };

        T6 = vector<float>{ 1, 2, 3 };

        T5 = vector<float>{ 10 };

        T3 = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
        T4 = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30 };

        Tensor T1({ 1, 2, 3, 1, 2 });
        T1 = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };

        A2 = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };
        B2 = vector<float>{ 1, 2, 3, 4, 5, 6 };
        A3 = vector<float>{ 1, 2 };
        B3 = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };

        Tensor Br = A2 + T6;
        Br.toCPU().print();
        cout << "\n(";
        for (int i = 0; i < Br.dim(); i++)
        {
            cout << Br.getShape()[i];
            if (i != Br.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\n\nTest\n\n";

        vector<int> v({ 4, 2 });
        vector<int> newv;
        newv.assign(2, 1);
        newv.insert(newv.end(), v.begin(), v.end());

        for (int i = 0; i < newv.size(); i++)
            cout << newv[i] << " ";

        cout << "\n\nPadding test\n\n";

        Tensor T14({ 3, 2, 4 });
        T14 = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };

        Tensor T15 = Tensor::pad(T9, 1);
        T15.toCPU().print();

        cout << "\n\nFlatten test\n\n";

        T14.flatten();
        T14.toCPU().print();
        cout << "\n(";
        for (int i = 0; i < T14.dim(); i++)
        {
            cout << T14.getShape()[i];
            if (i != T14.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\n\nConv2D test\n\n";

        Tensor T16({ 2, 3, 4, 4 });
        //{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48 };
        //{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
        T16 = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96 };

        CPUTensor::setSeed(42);
        Tensor K = Tensor::random({ 3, 3, 3, 3 });

        Tensor T17 = Tensor::conv2D(T16, K, 3, 2, 1, 1);
        T16.toCPU().print();
        cout << "\n";
        K.toCPU().print();
        cout << "\n";
        T17.toCPU().print();
        cout << "\n(";
        for (int i = 0; i < T17.dim(); i++)
        {
            cout << T17.getShape()[i];
            if (i != T17.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\n\nMax pool 2D test\n\n";

        MaxPoolRes mx_pl = Tensor::maxPool2D(T16, 2, 2, 2, 1);
        T16.toCPU().print();
        cout << "\n";
        mx_pl.vals.toCPU().print();
        cout << "\n(";
        for (int i = 0; i < mx_pl.vals.dim(); i++)
        {
            cout << mx_pl.vals.getShape()[i];
            if (i != mx_pl.vals.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\n\nConv init params test\n";

        vector<convStats> cv_stats = {
            {3, 16, 3, 1, 1, 0},
            {16, 32, 3, 1, 1, 0}
        };

        vector<MaxPoolStats> pl_stats = {
            {2, 2, 2, 0},
            {2, 1, 1, 0}
        };

        Tensor TX = Tensor::random({ 5000, 3, 28, 28 });
        Tensor TY = Tensor::random({ 5000, 10 });

        Parameters conv_params = init_conv(cv_stats, pl_stats, TX, TY.getShape()[1], "relu");

        conv_params.W[1].toCPU().print_dims();

        conv_params.W[2].toCPU().print_dims();

        conv_params.B[1].toCPU().print_dims();

        conv_params.B[2].toCPU().print_dims();

        conv_params.W[3].toCPU().print_dims();

        conv_params.B[3].toCPU().print_dims();

        cout << "\n\nInt32 data type test\n\n";

        Tensor Tcatva({ 2, 3 }, DataType::int32);
        Tcatva = vector<int32_t>{ 1, 2, 3, 4, 5, 6 };
        Tcatva.toCPU().print();

        cout << "\n\nFull CNN pass\n\n";
        CPUTensor::setSeed(42);
        Tensor Img = Tensor::randomUniform({ 4, 2, 8, 10 }, 0.0f, 1.0f);
        Tensor Y_Img({ 10, 4 });
        Y_Img = vector<float>{
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0
        };

        vector<convStats> cv_stats1 = {
            {2, 3, 3, 2, 1, 1},
            {3, 2, 2, 1, 1, 0}
        };

        vector<MaxPoolStats> mx_stats1 = {
            {2, 1, 2, 0},
            {2, 1, 1, 0}
        };

        Parameters cnn_params = init_conv(cv_stats1, mx_stats1, Img, 10, "relu");
        numerical_gradient_test(cnn_params, Img, Y_Img, cv_stats1, mx_stats1, "relu");
        AdamState cnn_adam1(cnn_params);
        Forward cnn_frd_cache = conv_frdpass(cnn_params, Img, cv_stats1, mx_stats1, "relu");
        Backward cnn_grads = conv_bckprop(cnn_frd_cache, cnn_params, Y_Img, cv_stats1, mx_stats1, "relu");
        auto printShape = [](const string& name, const Tensor& tensor)
            {
                const auto shape = tensor.getShape();

                cout << name << ": (";

                for (size_t i = 0; i < shape.size(); i++)
                {
                    if (i > 0)
                        cout << ", ";

                    cout << shape[i];
                }

                cout << ")\n";
            };

        printShape("Input", Img);
        printShape("Labels", Y_Img);

        int L = static_cast<int>(cnn_params.W.size());

        for (int l = 1; l < L; l++)
        {
            cout << "\nLayer " << l
                << (l == L - 1 ? " - Fully connected\n" : " - Convolution\n");

            printShape("W", cnn_params.W[l]);
            printShape("B", cnn_params.B[l]);

            printShape("Z", cnn_frd_cache.Z[l]);
            printShape("A", cnn_frd_cache.A[l]);

            if (l < L - 1)
                printShape("MaxPool indices", cnn_frd_cache.MPoolIdxs[l]);

            printShape("dZ", cnn_grads.dZ[l]);
            printShape("dW", cnn_grads.dW[l]);
            printShape("dB", cnn_grads.dB[l]);

            cout << "Weight gradient shape: "
                << (cnn_params.W[l].getShape() == cnn_grads.dW[l].getShape()
                    ? "MATCH" : "MISMATCH")
                << "\n";

            cout << "Bias gradient shape: "
                << (cnn_params.B[l].getShape() == cnn_grads.dB[l].getShape()
                    ? "MATCH" : "MISMATCH")
                << "\n";
        }
        adam(cnn_params, cnn_grads, cnn_adam1, 0.01f, 0.9f, 0.999f, 1e-8f);

        Tensor Imgtest = Img.clone();
        Tensor Y_Imgtest = Y_Img.clone();

        Parameters cnn_params_test1 = init_conv(cv_stats1, mx_stats1, Img, Y_Img.getShape()[0], "relu");
        AdamState cnn_state_test1(cnn_params_test1);
        CNN_model cnn_model_test1(cnn_params_test1, cv_stats1, mx_stats1, "relu", cnn_state_test1);

        train_cnn(cnn_model_test1, Img, Imgtest, Y_Img, Y_Imgtest, 0.001f, 1000, 100);

        cout << "\n\nSimple text Write + Read test";
        {
            ofstream out("test1.txt");

            if (!out)
                throw runtime_error("Unable to open file!");

            out << 2.5f << "\n";
            out << 23457.8f;

            if (!out)
                throw runtime_error("Unable to write to file!");

            out.close();

            if (!out)
                throw runtime_error("Unable to close file!");
        }

        float a = 0.0f;
        float b = 0.0f;

        {
            ifstream in("test1.txt");

            if (!in)
                throw runtime_error("Unable to open file!");

            if (!(in >> a >> b))
                throw runtime_error("Unable to read file!");
        }
        cout << "\n\n" << a << " " << b << "\n";

        cout << "\n\nByte write + read test\n\n";

        {
            ofstream outbin("test2.bin", ios::binary | ios::trunc); //binary + delete anything before this point in the file

            if (!outbin)
                throw runtime_error("Unable to open file!");

            outbin.write(reinterpret_cast<const char*>(&b), sizeof(b));

            if (!outbin)
                throw runtime_error("Unable to write to file!");
        }

        float c = 0.0f;

        {
            ifstream inbin("test2.bin", ios::binary);

            if (!inbin)
                throw runtime_error("Unable to open file!");

            if (!(inbin.read(reinterpret_cast<char*>(&c), sizeof(c))))
                throw runtime_error("Unable to read from file!");
        }
        cout << "\n" << c << "\n\n";

        cout << "\n\nVector bytes write + read test\n\n";

        vector<float> xv = { 1, 2, 3, 4, 5, 7, 9 };

        {
            ofstream outbin1("test2.bin", ios::binary | ios::trunc);

            if (!outbin1)
                throw runtime_error("Unable to open file!");

            outbin1.write(reinterpret_cast<const char*>(xv.data()), xv.size() * sizeof(float));

            if (!outbin1)
                throw runtime_error("Unable to write to file!");
        }

        vector<float> load_xv(xv.size());

        {
            ifstream inbin1("test2.bin", ios::binary);

            if (!inbin1)
                throw runtime_error("Unable to open file!");

            if (!(inbin1.read(reinterpret_cast<char*>(load_xv.data()), load_xv.size() * sizeof(float))))
                throw runtime_error("Unable to read from file!");
        }
        cout << "\n";
        for (int i = 0; i < load_xv.size(); i++)
            cout << load_xv[i] << " ";
        cout << "\n\n";

        cout << "\n\nFunction vector bytes write + read test\n\n";

        vector<float> yv = { 1, 1, 1, 7, 7, 7, 8 };

        filesystem::path save_path = "C:/Users/ASUS/Downloads/Fulcrum_save_tests/test.bin";
        save_vector_bin(yv, save_path);

        vector<float> load_yv(7);

        load_vector_bin(load_yv, "C:/Users/ASUS/Downloads/Fulcrum_save_tests/test.bin");

        cout << "\n";
        for (int i = 0; i < load_yv.size(); i++)
            cout << load_yv[i] << " ";
        cout << "\n\n";

        cout << "\n\nStruct bytes write + read test\n\n";

        DataType type = DataType::int32;
        float dtype = type == DataType::float32 ? 0.0f : 1.0f;

        {
            ofstream structout("test3.bin", ios::binary | ios::trunc);

            if (!structout)
                throw runtime_error("Unable to open write file!");

            const char magic[4] = { 'F', 'L', 'C', 'T' };

            float v = 0.01f;

            structout.write(magic, sizeof(magic));

            structout.write(reinterpret_cast<const char*>(&v), sizeof(float));

            structout.write(reinterpret_cast<const char*>(&dtype), sizeof(float));

            if (!structout)
                throw runtime_error("Unable to complete file write!");
        }

        {
            ifstream structin("test3.bin", ios::binary);

            if (!structin)
                throw runtime_error("Unable to open read file!");

            char magic[4] = { 0 };
            const char magic_test[4] = { 'F', 'L', 'C', 'T' };

            structin.read(magic, sizeof(magic));

            if (std::memcmp(magic, magic_test, 4) != 0)
                throw runtime_error("Ivalid file format!");

            float v_test = 0.01f;

            float v = 0.0f;

            structin.read(reinterpret_cast<char*>(&v), sizeof(float));

            if (v != v_test)
                throw runtime_error("Invalid version!");

            if (!(structin.read(reinterpret_cast<char*>(&dtype), sizeof(float))))
                throw runtime_error("Unable to read from file!");
        }
        if (dtype == 0.0f)
            cout << "\n\nfloat32" << "\n";
        if (dtype == 1.0f)
            cout << "\n\nint32" << "\n";

        cout << "\n\nTensor bytes write + read test!\n\n";

        Tensor WRT({ 2, 3 });
        WRT = vector<float>{ 1, 2, 3, 4, 5, 6 };
        WRT.toCPU().print();
        CPUTensor wrt_cpu = WRT.toCPU();

        const char magic[4] = { 'F', 'L', 'C', 'T' };
        float vr = 0.01f;

        {
            ofstream tout("test4.bin", ios::binary | ios::trunc);

            if (!tout)
                throw runtime_error("Unable to ope write file!");

            float dtype = wrt_cpu.getDataType() == DataType::float32 ? 0.0f : 1.0f;

            int rank = wrt_cpu.dim();

            tout.write(magic, sizeof(magic));

            tout.write(reinterpret_cast<const char*>(&vr), sizeof(vr));

            tout.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));

            tout.write(reinterpret_cast<const char*>(&rank), sizeof(rank));

            tout.write(reinterpret_cast<const char*>(wrt_cpu.getShape().data()), rank * sizeof(rank));

            tout.write(reinterpret_cast<const char*>(wrt_cpu.getFloatData()), wrt_cpu.size() * sizeof(float));

            if (!tout)
                throw runtime_error("Unable to finish file write!");
        }

        {
            ifstream tin("test4.bin", ios::binary);

            if (!tin)
                throw runtime_error("Unable to open read file!");

            char load_magic[4] = { 0 };

            tin.read(load_magic, sizeof(load_magic));

            if (memcmp(load_magic, magic, 4) != 0)
                throw runtime_error("Invalid file format!");

            float load_vr = 0.0f;

            tin.read(reinterpret_cast<char*>(&load_vr), sizeof(load_vr));

            if (load_vr != vr)
                throw runtime_error("Invalid file version!");

            float load_dtype = 0.0f;

            tin.read(reinterpret_cast<char*>(&load_dtype), sizeof(load_dtype));

            int load_rank = 0;

            tin.read(reinterpret_cast<char*>(&load_rank), sizeof(load_rank));

            vector<int> load_shape(load_rank);

            tin.read(reinterpret_cast<char*>(load_shape.data()), load_rank * sizeof(load_rank));

            Tensor load_wrt(load_shape);

            vector<float> load_payload(load_wrt.size());

            tin.read(reinterpret_cast<char*>(load_payload.data()), load_wrt.size() * sizeof(float));

            load_wrt = load_payload;

            cout << "\n";
            load_wrt.toCPU().print();
        }

        cout << "\n\nTensor bytes function write + read test\n\n";

        Tensor WRT1({ 4, 5 });
        WRT1 = vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };

        save_tensor(WRT1, "C:/Users/ASUS/Downloads/Fulcrum_save_tests/test1.bin");

        Tensor load_wrt1 = load_tensor("C:/Users/ASUS/Downloads/Fulcrum_save_tests/test1.bin");

        cout << "\n";
        load_wrt1.toCPU().print();

        cout << "\n\nModel bytes function write + read test\n\n";

        save_model(cnn_model_test1.params, "C:/Users/ASUS/Downloads/Fulcrum_save_tests/test2.bin");

        Parameters load_params = load_model("C:/Users/ASUS/Downloads/Fulcrum_save_tests/test2.bin");

        compare_params(cnn_model_test1.params, load_params);

        cout << "\n\nFull model bytes function write + read test!\n\n";

        save_cnn(cnn_model_test1, "C:/Users/ASUS/Downloads/Fulcrum_save_tests/test3.bin");

        CNN_model load_model1 = load_cnn("C:/Users/ASUS/Downloads/Fulcrum_save_tests/test3.bin");

        compare_cnn(cnn_model_test1, load_model1);

        cout << "\n\nJust some indexing test\n\n";

        CPUTensor one_img_cnn = choose_img(X_test_meteors, 7);

        cout << "\n(";
        for (int i = 0; i < one_img_cnn.dim(); i++)
        {
            cout << one_img_cnn.getShape()[i];
            if (i != one_img_cnn.dim() - 1)
                cout << ", ";
        }
        cout << ")";

        cout << "\n\nY print test\n\n";

        printMnistImage(X_test_mnist, 7);

        cout<<choose_label(y_test_mnist.toCUDA(), 7);

        cout << "\n\nPrediction test\n\n";

        predict_cnn(load_model1, Imgtest, Y_Imgtest, 2);

        save_cnn_image(X_test_meteors, 7, "C:/Users/ASUS/Downloads/selected_meteor.ppm");

        cout << "\n\nParams copy test\n\n";

        cout << load_model1.params.W[1].toCPU()(7) << "\n\n";

        change_params(load_model1);

        cout << load_model1.params.W[1].toCPU()(7) << "\n\n";

        cout << "\n\nParams copy test2\n\n";

        cout << load_model1.params.W[1].toCPU()(7) << "\n\n";

        change_params2(load_model1);

        cout << load_model1.params.W[1].toCPU()(7) << "\n\n";


        cout << "\n\nCUDA Meteors dataset test\n\n";

        vector<convStats> cv_meteors = {
            {3, 8, 3, 1, 1, 1},
            {8, 16, 3, 1, 1, 1},
            {16, 32, 3, 1, 1, 1}
        };

        vector<MaxPoolStats> mx_meteors = {
            {2, 2, 2, 0},
            {2, 2, 2, 0},
            {2, 2, 2, 0}
        };

        {
            cout << "Early stopping / restore best / target loss test\n\n";

            vector<convStats> cv_cnn = {
            {3, 8, 3, 1, 1, 1},
            {8, 16, 3, 1, 1, 1}
            };

            vector<MaxPoolStats> mx_cnn = {
                {2, 2, 2, 0},
                {2, 2, 2, 0}
            };

            start = chrono::high_resolution_clock::now();

            CPUTensor::setSeed(42);

            Parameters params_stopping = init_conv(cv_cnn, mx_cnn, X_train_cnn.toCUDA(), y_train_cnn.getShape()[0], "relu");
            AdamState state_stopping(params_stopping);
            CNN_model model_stopping(params_stopping, cv_cnn, mx_cnn, "relu", state_stopping);

            train_cnn(model_stopping, X_train_cnn.toCUDA(), X_test_cnn.toCUDA(), y_train_cnn.toCUDA(), y_test_cnn.toCUDA(), 0.001f, 120, 10, 0.0f, 0.0f, 0.1f, false, false, 0.001f, true, 3, 10, 1.70f, 0.01f);

            cudaDeviceSynchronize();

            Forward eval = conv_frdpass(model_stopping.params, X_test_cnn.toCUDA(), model_stopping.cv_stats, model_stopping.mx_stats, model_stopping.activ);

            float returned_cost = cost(y_test_cnn.toCUDA(), eval.A[model_stopping.params.W.size() - 1], model_stopping.params);

            cudaDeviceSynchronize();

            cout << "Returned model cost: " << returned_cost << '\n';

            end = chrono::high_resolution_clock::now();

            elapsed = end - start;

            cout << "\nEarly stopping training time: " << elapsed.count() << " seconds\n";

            predict_cnn(model_stopping, X_test_cnn.toCUDA(), y_test_cnn.toCUDA(), 7);

            cout << "\n\n";
        }

        {
            cout << "Best config train run\n\n";

            start = chrono::high_resolution_clock::now();

            CPUTensor::setSeed(42);

            Parameters params_best = init_conv(cv_meteors, mx_meteors, X_train_meteors.toCUDA(), y_train_meteors.getShape()[0], "relu");
            AdamState state_best(params_best);
            CNN_model model_best(params_best, cv_meteors, mx_meteors, "relu", state_best);

            train_cnn(model_best, X_train_meteors.toCUDA(), X_val_meteors.toCUDA(), y_train_meteors.toCUDA(), y_val_meteors.toCUDA(), 0.001f, 110, 10, 0.0f, 0.0f, 0.1f);

            cudaDeviceSynchronize();

            end = chrono::high_resolution_clock::now();

            elapsed = end - start;

            cout << "\nBest config Meteors training time: " << elapsed.count() << " seconds\n";

            predict_cnn(model_best, X_test_meteors.toCUDA(), y_test_meteors.toCUDA(), 7);

            cout << "\n\n";
        }

        {
            cout << "Test1: all 0\n\n";

            start = chrono::high_resolution_clock::now();

            CPUTensor::setSeed(42);

            Parameters params_test1 = init_conv(cv_meteors, mx_meteors, X_train_meteors.toCUDA(), y_train_meteors.getShape()[0], "relu");
            AdamState state1(params_test1);
            CNN_model model1(params_test1, cv_meteors, mx_meteors, "relu", state1);

            train_cnn(model1, X_train_meteors.toCUDA(), X_val_meteors.toCUDA(), y_train_meteors.toCUDA(), y_val_meteors.toCUDA(), 0.001f, 150, 17);

            cudaDeviceSynchronize();

            end = chrono::high_resolution_clock::now();

            elapsed = end - start;

            cout << "\nTest1 Meteors training time: " << elapsed.count() << " seconds\n";
        }

        {
            cout << "\nTest2: dropout = 0.2\n\n";

            start = chrono::high_resolution_clock::now();

            CPUTensor::setSeed(42);

            Parameters params_test2 = init_conv(cv_meteors, mx_meteors, X_train_meteors.toCUDA(), y_train_meteors.getShape()[0], "relu");
            AdamState state2(params_test2);
            CNN_model model2(params_test2, cv_meteors, mx_meteors, "relu", state2);

            train_cnn(model2, X_train_meteors.toCUDA(), X_val_meteors.toCUDA(), y_train_meteors.toCUDA(), y_val_meteors.toCUDA(), 0.001f, 150, 17, 0.2f);

            cudaDeviceSynchronize();

            end = chrono::high_resolution_clock::now();

            elapsed = end - start;

            cout << "\nTest2 Meteors training time: " << elapsed.count() << " seconds\n";
        }

        {
            cout << "\nTest3: L2 = 0.1\n\n";

            start = chrono::high_resolution_clock::now();

            CPUTensor::setSeed(42);

            Parameters params_test3 = init_conv(cv_meteors, mx_meteors, X_train_meteors.toCUDA(), y_train_meteors.getShape()[0], "relu");
            AdamState state3(params_test3);
            CNN_model model3(params_test3, cv_meteors, mx_meteors, "relu", state3);

            train_cnn(model3, X_train_meteors.toCUDA(), X_val_meteors.toCUDA(), y_train_meteors.toCUDA(), y_val_meteors.toCUDA(), 0.001f, 150, 17, 0.0f, 0.0f, 0.1f);

            cudaDeviceSynchronize();

            end = chrono::high_resolution_clock::now();

            elapsed = end - start;

            cout << "\nTest3 Meteors training time: " << elapsed.count() << " seconds\n";
        }

        {
            cout << "\nTest4: dropout = 0.2 + L2 = 0.1\n\n";

            start = chrono::high_resolution_clock::now();

            CPUTensor::setSeed(42);

            Parameters params_test4 = init_conv(cv_meteors, mx_meteors, X_train_meteors.toCUDA(), y_train_meteors.getShape()[0], "relu");
            AdamState state4(params_test4);
            CNN_model model4(params_test4, cv_meteors, mx_meteors, "relu", state4);

            train_cnn(model4, X_train_meteors.toCUDA(), X_val_meteors.toCUDA(), y_train_meteors.toCUDA(), y_val_meteors.toCUDA(), 0.001f, 150, 17, 0.2f, 0.0f, 0.1f);

            cudaDeviceSynchronize();

            end = chrono::high_resolution_clock::now();

            elapsed = end - start;

            cout << "\nTest4 Meteors training time: " << elapsed.count() << " seconds\n";
        }

        {
            cout << "\nTest5: tanh all 0\n\n";

            start = chrono::high_resolution_clock::now();

            CPUTensor::setSeed(42);

            Parameters params_test5 = init_conv(cv_meteors, mx_meteors, X_train_meteors.toCUDA(), y_train_meteors.getShape()[0], "tanh");
            AdamState state5(params_test5);
            CNN_model model5(params_test5, cv_meteors, mx_meteors, "tanh", state5);

            train_cnn(model5, X_train_meteors.toCUDA(), X_val_meteors.toCUDA(), y_train_meteors.toCUDA(), y_val_meteors.toCUDA(), 0.001f, 150, 17);

            cudaDeviceSynchronize();

            end = chrono::high_resolution_clock::now();

            elapsed = end - start;

            cout << "\nTest5 Meteors training time: " << elapsed.count() << " seconds\n";
        }

        {
            cout << "\nTest6: tanh dropout = 0.2\n\n";

            start = chrono::high_resolution_clock::now();

            CPUTensor::setSeed(42);

            Parameters params_test6 = init_conv(cv_meteors, mx_meteors, X_train_meteors.toCUDA(), y_train_meteors.getShape()[0], "tanh");
            AdamState state6(params_test6);
            CNN_model model6(params_test6, cv_meteors, mx_meteors, "tanh", state6);

            train_cnn(model6, X_train_meteors.toCUDA(), X_val_meteors.toCUDA(), y_train_meteors.toCUDA(), y_val_meteors.toCUDA(), 0.001f, 150, 17, 0.2f);

            cudaDeviceSynchronize();

            end = chrono::high_resolution_clock::now();

            elapsed = end - start;

            cout << "\nTest6 Meteors training time: " << elapsed.count() << " seconds\n";
        }

        {
            cout << "\nTest7: tanh L2 = 0.1\n\n";

            start = chrono::high_resolution_clock::now();

            CPUTensor::setSeed(42);

            Parameters params_test7 = init_conv(cv_meteors, mx_meteors, X_train_meteors.toCUDA(), y_train_meteors.getShape()[0], "tanh");
            AdamState state7(params_test7);
            CNN_model model7(params_test7, cv_meteors, mx_meteors, "tanh", state7);

            train_cnn(model7, X_train_meteors.toCUDA(), X_val_meteors.toCUDA(), y_train_meteors.toCUDA(), y_val_meteors.toCUDA(), 0.001f, 150, 17, 0.0f, 0.0f, 0.1f);

            cudaDeviceSynchronize();

            end = chrono::high_resolution_clock::now();

            elapsed = end - start;

            cout << "\nTest7 Meteors training time: " << elapsed.count() << " seconds\n";
        }

        {
            cout << "\nTest8: tanh dropout = 0.2 + L2 = 0.1\n\n";

            start = chrono::high_resolution_clock::now();

            CPUTensor::setSeed(42);

            Parameters params_test8 = init_conv(cv_meteors, mx_meteors, X_train_meteors.toCUDA(), y_train_meteors.getShape()[0], "tanh");
            AdamState state8(params_test8);
            CNN_model model8(params_test8, cv_meteors, mx_meteors, "tanh", state8);

            train_cnn(model8, X_train_meteors.toCUDA(), X_val_meteors.toCUDA(), y_train_meteors.toCUDA(), y_val_meteors.toCUDA(), 0.001f, 150, 17, 0.2f, 0.0f, 0.1f);

            cudaDeviceSynchronize();

            end = chrono::high_resolution_clock::now();

            elapsed = end - start;

            cout << "\nTest8 Meteors training time: " << elapsed.count() << " seconds\n";
        }
        

        cout << "\n\nCUDA cifar10 dataset test\n\n";

        start = chrono::high_resolution_clock::now();

        CPUTensor::setSeed(42);

        vector<convStats> cv_cnn = {
            {3, 8, 3, 1, 1, 1},
            {8, 16, 3, 1, 1, 1}
        };

        vector<MaxPoolStats> mx_cnn = {
            {2, 2, 2, 0},
            {2, 2, 2, 0}
        };

        Parameters cnn_params_cifar10 = init_conv(cv_cnn, mx_cnn, X_train_cnn.toCUDA(), y_train_cnn.getShape()[0], "relu");
        AdamState cnn_state_cifar10(cnn_params_cifar10);
        CNN_model cnn_model_cifar10(cnn_params_cifar10, cv_cnn, mx_cnn, "relu", cnn_state_cifar10);

        train_cnn(cnn_model_cifar10, X_train_cnn.toCUDA(), X_test_cnn.toCUDA(), y_train_cnn.toCUDA(), y_test_cnn.toCUDA(), 0.001f, 1000, 100);

        cudaDeviceSynchronize();

        end = chrono::high_resolution_clock::now();

        elapsed = end - start;

        cout << "\nCUDA cifar10 training time: " << elapsed.count() << " seconds\n";


        cout << "\n\nCUDA cat dataset test\n\n";

        start = std::chrono::high_resolution_clock::now();

        CPUTensor::setSeed(42);

        Parameters params3 = train(X_train_cat.toCUDA(), X_test_cat.toCUDA(), y_train_cat.toCUDA(), y_test_cat.toCUDA(), dim_list, "tanh", 0.0001f, 700, 100, 0.9f, 0.999f, 1e-8f, 0.2f, 0.0f, 0.01f);

        cudaDeviceSynchronize();

        end = std::chrono::high_resolution_clock::now();

        elapsed = end - start;

        cout << "\nCUDA cat training time: " << elapsed.count() << " seconds\n";

        predict(X_test_cat, y_test_cat, params3, "tanh", 7);


        dim_list = { X_train_mnist.getShape()[X_train_cat.dim() - 2], 100, 100, 200, y_train_mnist.getShape()[y_train_mnist.dim() - 2] };


        cout << "\n\nCUDA mnist dataset test\n\n";

        start = std::chrono::high_resolution_clock::now();

        CPUTensor::setSeed(123);

        Parameters params6 = train(X_train_mnist.toCUDA(), X_test_mnist.toCUDA(), y_train_mnist.toCUDA(), y_test_mnist.toCUDA(), dim_list, "relu", 0.0001f, 1000, 100, 0.9f, 0.999f, 1e-8f, 0.0f, 0.0f, 0.01f);

        cudaDeviceSynchronize();

        end = std::chrono::high_resolution_clock::now();

        elapsed = end - start;

        cout << "\nCUDA mnist training time: " << elapsed.count() << " seconds\n";
        
        predict(X_test_mnist, y_test_mnist, params6, "relu", 101);
    }
    catch (const exception& e)
    {
        cerr << "Error: " << e.what() << "\n";
    }

    return 0;
}