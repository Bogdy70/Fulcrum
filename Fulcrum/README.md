# Fulcrum

Fulcrum is a Windows C++/CUDA project exploring GPU-backed tensor operations and neural-network training. It contains a CUDA tensor implementation, a CPU tensor/data-loading utility, and example fully connected and convolutional classifiers. The current executable is a set of demonstrations and training experiments rather than a configurable command-line application or reusable packaged library.

## Capabilities

- CPU tensors and CUDA device tensors with `float32` and `int32` storage, shape/stride metadata, and CPU/GPU data transfers.
- Tensor creation, reshape/resize, squeeze/unsqueeze/flatten, padding, elementwise arithmetic and comparisons, broadcasting, matrix multiplication, reductions, and common math/activation operations.
- CUDA 2D convolution and max-pooling operations, including backward-operation kernels used by convolutional training.
- Fully connected and convolutional neural-network training code with ReLU or tanh hidden activations, Adam optimization, dropout, and optional L1/L2 regularization.
- Binary dataset loading and sample prediction/printing helpers.

The implementation is in `Tensor.cu`/`Tensor.cuh`, `CPUTensor.cpp`/`CPUTensor.h`, and `Fulcrum.cpp`. The current `main` loads all bundled datasets, prints tensor demonstrations, and runs multiple CUDA training experiments. It is expected to take a while to finish; it is not a short startup or unit-test command. No separate test project or command-line options are currently provided.

## Requirements

- Windows and Visual Studio/MSBuild with the C++ desktop development workload and the MSVC `v145` platform toolset used by the project.
- CUDA Toolkit 13.2, including its Visual Studio build customizations/integration.
- An NVIDIA CUDA-capable GPU. The x64 project configurations compile CUDA for `compute_89` / `sm_89`; other GPU architectures may require changing the CUDA code-generation setting in `Fulcrum.vcxproj` and rebuilding.
- The Windows 10 SDK selected by the project (`10.0`).

The project uses C++20. Its x64 Release configuration enables OpenMP support; CUDA is configured in the x64 Debug and Release configurations. The checked-in Win32 configurations do not include the same CUDA code-generation settings, so x64 is the intended configuration for the CUDA demonstrations.

## Build and run

1. Open `Fulcrum.vcxproj` in Visual Studio.
2. Select **x64** and **Release** (or **Debug**) in the configuration toolbar.
3. Build the `Fulcrum` project.
4. Run the resulting console application with `Fulcrum/` as its working directory.

The program uses paths such as `data/cat/X_train.bin` relative to its current working directory. In Visual Studio, set **Project Properties → Configuration Properties → Debugging → Working Directory** to `$(ProjectDir)` if the application cannot find the data files. When launching outside Visual Studio, change into the `Fulcrum` directory before starting the executable.

All dataset files listed below are loaded at the start of `main`, even when you only want to work on one model. Keep the complete `data/` directory in place or update the paths and loader calls in `Fulcrum.cpp`. Errors loading the data are printed to standard error.

## Bundled datasets

The `.bin` files contain raw little-endian IEEE-754 32-bit floating-point values, with no header. Shapes are supplied by the loader in `Fulcrum.cpp`; data is interpreted in row-major order. Matrix-style datasets store examples in columns. Image CNN inputs use `[N, C, H, W]` shapes.

| Dataset | Files | Shapes used by the program |
| --- | --- | --- |
| Cat binary classification | `data/cat/{X_train,Y_train,X_test,Y_test}.bin` | `X_train`: `[12288, 209]`; `Y_train`: `[1, 209]`; `X_test`: `[12288, 50]`; `Y_test`: `[1, 50]` |
| MNIST digit classification | `data/mnist/{X_train,Y_train,X_test,Y_test}.bin` | `X_train`: `[784, 5000]`; `Y_train`: `[10, 5000]`; `X_test`: `[784, 1000]`; `Y_test`: `[10, 1000]` |
| CIFAR-10 CNN experiment | `data/cifar10_cnn/{X_train,Y_train,X_test,Y_test}.bin` | `X_train`: `[1000, 3, 32, 32]`; `Y_train`: `[10, 1000]`; `X_test`: `[100, 3, 32, 32]`; `Y_test`: `[10, 100]` |
| Meteor binary CNN experiments | `data/meteor_cnn/{X_train,Y_train,X_val,Y_val,X_test,Y_test}.bin` | `X_train`: `[1000, 3, 128, 128]`; `Y_train`: `[1, 1000]`; validation/test images: `[100, 3, 128, 128]`; validation/test labels: `[1, 100]` |

`data/cat/cat_shapes.txt` and `data/mnist/shapes.txt` record the corresponding matrix dimensions. The loader does not define image normalization or label encoding; if you replace a dataset, prepare values and labels in a format compatible with the loss and accuracy routines in `Fulcrum.cpp`, and keep the shapes passed to the binary loaders in sync with the files.

`CPUTensor::loadTensorBin` checks that the file size exactly matches the requested shape and requires a little-endian host. `loadMatrixBin` reads the requested number of floats from the file using the caller-supplied dimensions.

## Source layout

```text
Fulcrum/
├── Fulcrum.vcxproj       Visual Studio project and CUDA build configuration
├── Fulcrum.cpp           Neural-network code, experiments, and program entry point
├── Tensor.cuh            CUDA tensor API
├── Tensor.cu             CUDA storage and operation kernels
├── CPUTensor.h/.cpp      Host tensor API and binary dataset loaders
└── data/                 Cat, MNIST, CIFAR-10, and meteor sample binaries
```

## Notes and limitations

- The executable's datasets, dimensions, model settings, and experiment sequence are currently hard-coded in `Fulcrum.cpp`; there are no command-line flags or configuration files.
- Running the full program requires every dataset referenced at the beginning of `main`, including meteor and CIFAR-10 data, and starts several potentially long training runs.
- The CUDA build targets `sm_89`. To use a different GPU generation, update the CUDA code-generation setting and ensure the installed CUDA Toolkit supports the desired target.
- This repository does not currently include separate automated tests, installation packaging, or instructions for generating/replacing the binary datasets.
