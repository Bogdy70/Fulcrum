#include <fulcrum/nn/NN.h>
#include <fulcrum/data/Data.h>
#include <iostream>

using namespace std;
using namespace fulcrum::data;

namespace fulcrum::nn
{
    float cost(const Tensor& Y, const Tensor& pred, const Parameters& params, const float lambda_l1, const float lambda_l2)
    {
        float m = static_cast<float>(Y.getShape()[Y.dim() - 1]);
        int L = size(params.W);
        float epsilon = 1e-7f;
        float cost = 0.0f;
        float sumw = 0.0f;
        Tensor clippedPred = Tensor::clipT(pred, epsilon, 1.0f - epsilon);

        if (lambda_l1 < 0.0f || lambda_l2 < 0.0f)
            throw runtime_error("Lambda value cannot be less than zero.");

        if (Y.getShape()[Y.dim() - 2] > 1)
            cost = (-1.0f / m) * Tensor::sum(Y * Tensor::logT(clippedPred)).toScalar();
        else
            cost = (-1.0f / m) * Tensor::sum(Y * Tensor::logT(clippedPred) + (1 - Y) * Tensor::logT(1 - clippedPred)).toScalar();

        if (lambda_l1 > 0.0f)
        {
            sumw = 0.0f;
            for (int l = 1; l < L; l++)
            {
                sumw += Tensor::sum(Tensor::absT(params.W[l])).toScalar();
            }
            cost += (lambda_l1 / m) * sumw;
        }
        if (lambda_l2 > 0.0f)
        {
            sumw = 0.0f;
            for (int l = 1; l < L; l++)
            {
                sumw += Tensor::sum(params.W[l] * params.W[l]).toScalar();
            }
            cost += (lambda_l2 / (2.0f * m)) * sumw;
        }
        return cost;
    }

    float accuracy(const Tensor& Y, const Tensor& pred)
    {
        if (Y.getShape()[Y.dim() - 2] > 1)
        {
            Tensor true_labels = Tensor::argmax(Y, 0);
            Tensor pred_labels = Tensor::argmax(pred, 0);

            return Tensor::sum(true_labels == pred_labels).toScalar() / static_cast<float>(Y.getShape()[Y.dim() - 1]);
        }
        else
        {
            Tensor pred_labels = pred > 0.5f;
            return Tensor::sum(Y == pred_labels).toScalar() / static_cast<float>(Y.getShape()[Y.dim() - 1]);
        }
    }

    Parameters init_params(const vector<int>& dim_list)
    {
        int L = size(dim_list);
        Parameters params(L);

        for (int l = 1; l < L; l++)
        {
            params.W[l] = Tensor::random({ dim_list[l], dim_list[l - 1] }) * sqrt(2.0f / static_cast<float>(dim_list[l - 1]));
            params.B[l] = Tensor::zeros({ dim_list[l], 1 });
        }

        return params;
    }

    Parameters init_conv(const vector<convStats>& conv_stats, const vector<MaxPoolStats>& pool_stats, const Tensor& X, int output_size, const string& activation)
    {
        if (conv_stats.empty())
            throw runtime_error("At least one convolution layer is required!");

        if (conv_stats.size() != pool_stats.size())
            throw runtime_error("Every conv layer must have a maxpool layer!");

        if (output_size < 1)
            throw runtime_error("Invalid output size!");

        int L = size(conv_stats);

        Parameters params(L + 2);

        int N = X.getShape()[X.dim() - 2];
        int M = X.getShape()[X.dim() - 1];

        for (int l = 1; l <= L; l++)
        {
            float n_in = static_cast<float>(conv_stats[l - 1].in_channels * conv_stats[l - 1].kernel_size * conv_stats[l - 1].kernel_size);
            float n_out = static_cast<float>(conv_stats[l - 1].out_channels * conv_stats[l - 1].kernel_size * conv_stats[l - 1].kernel_size);

            float init = activation == "relu" ? sqrt(2.0f / n_in) : sqrt(2.0f / (n_in + n_out));

            params.W[l] = Tensor::random({ conv_stats[l - 1].out_channels, conv_stats[l - 1].in_channels, conv_stats[l - 1].kernel_size, conv_stats[l - 1].kernel_size }) * init;
            params.B[l] = Tensor::zeros({ conv_stats[l - 1].out_channels, 1, 1 });

            if (conv_stats[l - 1].kernel_size > N + 2 * conv_stats[l - 1].padding || conv_stats[l - 1].kernel_size > M + 2 * conv_stats[l - 1].padding)
                throw runtime_error("Invalid convolution kernel size!");

            if (pool_stats[l - 1].kernel_size > N + 2 * pool_stats[l - 1].padding || pool_stats[l - 1].kernel_size > M + 2 * pool_stats[l - 1].padding)
                throw runtime_error("Invalid max pool kernel size!");

            if (conv_stats[l - 1].hStride < 1)
                throw runtime_error("Invalid convolution horizontal stride value!");

            if (pool_stats[l - 1].hStride < 1)
                throw runtime_error("Invalid max pool horizontal stride value!");

            if (conv_stats[l - 1].vStride < 1)
                throw runtime_error("Invalid convolution vertical stride value!");

            if (pool_stats[l - 1].vStride < 1)
                throw runtime_error("Invalid max pool vertical stride value!");

            N = (N + 2 * conv_stats[l - 1].padding - conv_stats[l - 1].kernel_size) / conv_stats[l - 1].vStride + 1;
            N = (N + 2 * pool_stats[l - 1].padding - pool_stats[l - 1].kernel_size) / pool_stats[l - 1].vStride + 1;
            M = (M + 2 * conv_stats[l - 1].padding - conv_stats[l - 1].kernel_size) / conv_stats[l - 1].hStride + 1;
            M = (M + 2 * pool_stats[l - 1].padding - pool_stats[l - 1].kernel_size) / pool_stats[l - 1].hStride + 1;
        }

        if (N <= 0 || M <= 0)
            throw runtime_error("Invalid final shape!");

        int flat_shape = params.W[L].getShape()[0] * N * M;
        params.W[L + 1] = Tensor::random({ output_size, flat_shape }) * sqrt(2.0f / static_cast<float>(flat_shape + output_size));
        params.B[L + 1] = Tensor::zeros({ output_size, 1 });

        return params;
    }

    Forward forward_pass(const Parameters& params, const Tensor& X, const string& activation, const float dropout)
    {
        if (dropout < 0.0f || dropout >= 1.0f)
            throw runtime_error("Dropout value must be between [0, 1).");

        int L = size(params.W);
        Forward forward_cache(L);
        Activation activ(activation);
        Tensor(*final_activ)(const Tensor&);

        forward_cache.A[0] = X.clone();

        for (int l = 1; l < L - 1; l++)
        {
            forward_cache.Z[l] = params.W[l].matmul(forward_cache.A[l - 1]) + params.B[l];
            forward_cache.A[l] = activ.forward(forward_cache.Z[l]);
            if (dropout > 0.0f)
            {
                forward_cache.D[l] = Tensor::randomUniform({ forward_cache.A[l].getShape()[forward_cache.A[l].dim() - 2], forward_cache.A[l].getShape()[forward_cache.A[l].dim() - 1] }, 0.0f, 1.0f) < (1.0f - dropout);
                forward_cache.A[l] = forward_cache.A[l] * forward_cache.D[l] / (1.0f - dropout);
            }
        }
        forward_cache.Z[L - 1] = params.W[L - 1].matmul(forward_cache.A[L - 2]) + params.B[L - 1];
        final_activ = forward_cache.Z[L - 1].getShape()[forward_cache.Z[L - 1].dim() - 2] > 1 ? Tensor::softmax : Tensor::sigmoid;
        forward_cache.A[L - 1] = final_activ(forward_cache.Z[L - 1]);

        return forward_cache;
    }

    Forward conv_frdpass(const Parameters& params, const Tensor& X, const vector<convStats>& cv_stats, const vector<MaxPoolStats>& mx_stats, const string& activation, float dropout)
    {
        if (dropout < 0.0f || dropout >= 1.0f)
            throw runtime_error("Invalid dropout value!");

        int L = size(params.W);

        Forward frd_cache(L);
        Activation activ(activation);
        Tensor(*final_activ)(const Tensor&);

        frd_cache.A[0] = X.clone();

        for (int l = 1; l < L - 1; l++)
        {
            frd_cache.Z[l] = Tensor::conv2D(frd_cache.A[l - 1], params.W[l], cv_stats[l - 1].kernel_size, cv_stats[l - 1].hStride, cv_stats[l - 1].vStride, cv_stats[l - 1].padding) + params.B[l];
            frd_cache.A[l] = activ.forward(frd_cache.Z[l]);
            MaxPoolRes max_pool = Tensor::maxPool2D(frd_cache.A[l], mx_stats[l - 1].kernel_size, mx_stats[l - 1].hStride, mx_stats[l - 1].vStride, mx_stats[l - 1].padding);
            frd_cache.A[l] = move(max_pool.vals);
            frd_cache.MPoolIdxs[l] = move(max_pool.idxs);

            if (dropout > 0.0f)
            {
                frd_cache.D[l] = Tensor::randomUniform(frd_cache.A[l].getShape(), 0.0f, 1.0f) < (1.0f - dropout);
                frd_cache.A[l] = frd_cache.A[l] * frd_cache.D[l] / (1.0f - dropout);
            }
        }

        Tensor flat_A = frd_cache.A[L - 2].clone().flatten(1).T();
        frd_cache.Z[L - 1] = params.W[L - 1].matmul(flat_A) + params.B[L - 1];
        final_activ = frd_cache.Z[L - 1].getShape()[0] > 1 ? Tensor::softmax : Tensor::sigmoid;
        frd_cache.A[L - 1] = final_activ(frd_cache.Z[L - 1]);

        return frd_cache;
    }

    Backward backpropagation(const Forward& frd_cache, const Parameters& params, const Tensor& Y, const string& activation, const float dropout, const float lambda_l1, const float lambda_l2)
    {
        int L = size(frd_cache.A);
        float m = static_cast<float>(Y.getShape()[Y.dim() - 1]);
        Backward grads(L);
        Activation activ(activation);

        grads.dZ[L - 1] = frd_cache.A[L - 1] - Y;
        grads.dW[L - 1] = (1.0f / m) * grads.dZ[L - 1].matmul(frd_cache.A[L - 2].T());
        grads.dB[L - 1] = (1.0f / m) * Tensor::sum(grads.dZ[L - 1], 1, true);
        if (lambda_l1 > 0.0f)
            grads.dW[L - 1] = grads.dW[L - 1] + ((lambda_l1 / m) * Tensor::sign(params.W[L - 1]));
        if (lambda_l2 > 0.0f)
            grads.dW[L - 1] = grads.dW[L - 1] + ((lambda_l2 / m) * params.W[L - 1]);

        for (int l = L - 2; l > 0; l--)
        {
            grads.dZ[l] = params.W[l + 1].T().matmul(grads.dZ[l + 1]);
            if (dropout > 0.0f)
                grads.dZ[l] = grads.dZ[l] * frd_cache.D[l] / (1.0f - dropout);
            grads.dZ[l] = grads.dZ[l] * activ.derivate(frd_cache.Z[l]);
            grads.dW[l] = (1.0f / m) * grads.dZ[l].matmul(frd_cache.A[l - 1].T());
            if (lambda_l1 > 0.0f)
                grads.dW[l] = grads.dW[l] + ((lambda_l1 / m) * Tensor::sign(params.W[l]));
            if (lambda_l2 > 0.0f)
                grads.dW[l] = grads.dW[l] + ((lambda_l2 / m) * params.W[l]);
            grads.dB[l] = (1.0f / m) * Tensor::sum(grads.dZ[l], 1, true);
        }

        return grads;
    }

    Backward conv_bckprop(const Forward& frd_cache, const Parameters& params, const Tensor& Y, const vector<convStats>& cv_stats, const vector<MaxPoolStats>& mx_stats, const string activation, float dropout, float lambda_l1, float lambda_l2)
    {
        if (dropout < 0.0f || dropout >= 1.0f)
            throw runtime_error("Invalid dropout value!");

        if (lambda_l1 < 0.0f || lambda_l2 < 0.0f)
            throw runtime_error("Lambda values cannot be less than 0!");

        int L = size(frd_cache.A);
        float m = static_cast<float>(frd_cache.A[0].getShape()[0]);
        Backward grads(L);
        Activation activ(activation);

        grads.dZ[L - 1] = frd_cache.A[L - 1] - Y;
        Tensor flat_A = frd_cache.A[L - 2].clone();
        grads.dW[L - 1] = (1.0f / m) * grads.dZ[L - 1].matmul(flat_A.flatten(1));
        if (lambda_l1 > 0.0f)
            grads.dW[L - 1] = grads.dW[L - 1] + ((lambda_l1 / m) * Tensor::sign(params.W[L - 1]));
        if (lambda_l2 > 0.0f)
            grads.dW[L - 1] = grads.dW[L - 1] + ((lambda_l2 / m) * params.W[L - 1]);
        grads.dB[L - 1] = (1.0f / m) * Tensor::sum(grads.dZ[L - 1], 1, true);
        grads.dZ[L - 2] = params.W[L - 1].T().matmul(grads.dZ[L - 1]); //df
        grads.dZ[L - 2] = grads.dZ[L - 2].T();
        grads.dZ[L - 2].reshape(frd_cache.A[L - 2].getShape()); //dP

        for (int l = L - 2; l >= 1; l--)
        {
            if (dropout > 0.0f)
                grads.dZ[l] = grads.dZ[l] * frd_cache.D[l] / (1.0f - dropout);
            grads.dZ[l] = Tensor::back_maxPool2D(grads.dZ[l], frd_cache.MPoolIdxs[l], frd_cache.Z[l].getShape()[frd_cache.Z[l].dim() - 2], frd_cache.Z[l].getShape()[frd_cache.Z[l].dim() - 1], mx_stats[l - 1].padding); //dA
            grads.dZ[l] = grads.dZ[l] * activ.derivate(frd_cache.Z[l]); //dZ
            grads.dW[l] = Tensor::conv2D_dK(frd_cache.A[l - 1], grads.dZ[l], cv_stats[l - 1].kernel_size, cv_stats[l - 1].hStride, cv_stats[l - 1].vStride, cv_stats[l - 1].padding);
            if (lambda_l1 > 0.0f)
                grads.dW[l] = grads.dW[l] + ((lambda_l1 / m) * Tensor::sign(params.W[l]));
            if (lambda_l2 > 0.0f)
                grads.dW[l] = grads.dW[l] + ((lambda_l2 / m) * params.W[l]);
            grads.dB[l] = (1.0f / m) * Tensor::sum(Tensor::sum(Tensor::sum(grads.dZ[l], 3, true), 2, true), 0);
            if (l != 1)
                grads.dZ[l - 1] = Tensor::conv2D_dX(grads.dZ[l], params.W[l], frd_cache.A[l - 1].getShape()[frd_cache.A[l - 1].dim() - 2], frd_cache.A[l - 1].getShape()[frd_cache.A[l - 1].dim() - 1], cv_stats[l - 1].hStride, cv_stats[l - 1].vStride, cv_stats[l - 1].padding);
        }

        return grads;
    }

    Parameters train(const Tensor& X_train,
        const Tensor& X_test,
        const Tensor& y_train,
        const Tensor& y_test,
        const vector<int>& dim_list,
        const string& activation,
        const float lr,
        const int epochs,
        const int viewing_rate,
        const float beta1,
        const float beta2,
        const float eps,
        const float dropout,
        const float lambda_l1,
        const float lambda_l2)
    {
        Parameters params = init_params(dim_list);
        AdamState state(params);

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int epoch = 0; epoch <= epochs; epoch++)
        {
            if (epoch % viewing_rate == 0)
            {
                cudaDeviceSynchronize();

                Forward train_eval_cache = forward_pass(params, X_train, activation);
                Forward frd_cache_test = forward_pass(params, X_test, activation);

                float train_objective = cost(y_train, train_eval_cache.A[size(dim_list) - 1], params, lambda_l1, lambda_l2);
                float train_cost = cost(y_train, train_eval_cache.A[size(dim_list) - 1], params);
                float train_acc = accuracy(y_train, train_eval_cache.A[size(dim_list) - 1]);

                float test_cost = cost(y_test, frd_cache_test.A[size(dim_list) - 1], params);
                float test_acc = accuracy(y_test, frd_cache_test.A[size(dim_list) - 1]);

                auto current_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = current_time - start_time;

                cout << "Epoch: " << epoch << " || Train objective: " << train_objective << " || Train loss : " << train_cost << " || Test loss : " << test_cost << " || Train accuracy : " << train_acc * 100.0f << " % || Test accuracy : " << test_acc * 100.0f << " % || Time : " << elapsed.count() << " sec\n";
            }

            if (epoch == epochs)
                break;

            Forward frd_cache = forward_pass(params, X_train, activation, dropout);
            Backward grads = backpropagation(frd_cache, params, y_train, activation, dropout, lambda_l1, lambda_l2);
            //optimizer(params, grads, lr);
            adam(params, grads, state, lr, beta1, beta2, eps);
        }

        return params;
    }

    CNN_model& train_cnn(CNN_model& model,
        const Tensor& Xtrain,
        const Tensor& Xtest,
        const Tensor& ytrain,
        const Tensor& ytest,
        float lr,
        int epochs,
        int viewing_rate,
        float dropout,
        float lambda_l1,
        float lambda_l2,
        bool restore_best,
        bool restore_meaningful,
        float min_delta,
        bool early_stopping,
        int patience,
        int min_epochs,
        float target_loss,
        float target_tolerance,
        float beta1,
        float beta2,
        float eps)
    {
        auto start_time = chrono::high_resolution_clock::now();

        if (early_stopping != 0)
        {
            if (min_epochs > epochs)
                throw runtime_error("The minimum nimber of epochs cannot be higher than the maximum number!");

            if (restore_best != 0 && restore_meaningful != 0)
                throw runtime_error("Only one restore option can be selected!");
        }

        float reference_cost = numeric_limits<float>::infinity();
        float reference_cost_best = numeric_limits<float>::infinity();
        Parameters best_params(model.params.W.size());
        int counter = 0;
        bool snapshot = false;

        for (int epoch = 0; epoch <= epochs; epoch++)
        {
            if (restore_best != 0 || early_stopping != 0)
            {
                Forward test_frd_cache = conv_frdpass(model.params, Xtest, model.cv_stats, model.mx_stats, model.activ);
                float test_cost = cost(ytest, test_frd_cache.A[size(model.params.W) - 1], model.params);

                if (restore_best != 0 && target_loss == 0 && test_cost <= reference_cost_best)
                {
                    reference_cost_best = test_cost;
                    best_params = clone_params(model.params);
                    snapshot = true;
                }
                if (early_stopping != 0 && epoch >= min_epochs)
                {
                    if (target_loss != 0)
                    {
                        if (test_cost <= target_loss + target_tolerance)
                            break;
                    }
                    else
                    {
                        if (test_cost < reference_cost - min_delta)
                        {
                            counter = 0;
                            reference_cost = test_cost;
                            if (restore_meaningful != 0)
                            {
                                best_params = clone_params(model.params);
                                snapshot = true;
                            }
                        }
                        else
                        {
                            counter++;
                            if (counter >= patience)
                            {
                                cout << "Epoch: " << epoch << " || Cost: " << test_cost << " || Ref: " << reference_cost << " || Counter: " << counter << "\n";
                                break;
                            }
                        }
                    }
                }
            }

            if (epoch % viewing_rate == 0 || epoch == epochs)
            {
                cudaDeviceSynchronize();

                Forward train_frd_cache = conv_frdpass(model.params, Xtrain, model.cv_stats, model.mx_stats, model.activ);
                Forward test_frd_cache = conv_frdpass(model.params, Xtest, model.cv_stats, model.mx_stats, model.activ);

                float train_objective = cost(ytrain, train_frd_cache.A[size(model.params.W) - 1], model.params, lambda_l1, lambda_l2);
                float train_cost = cost(ytrain, train_frd_cache.A[size(model.params.W) - 1], model.params);
                float train_accuracy = accuracy(ytrain, train_frd_cache.A[size(model.params.W) - 1]);

                float test_cost = cost(ytest, test_frd_cache.A[size(model.params.W) - 1], model.params);
                float test_accuracy = accuracy(ytest, test_frd_cache.A[size(model.params.W) - 1]);

                auto current_time = chrono::high_resolution_clock::now();
                chrono::duration<double> time = current_time - start_time;

                cout << "Epoch: " << epoch << " || Train objective: " << train_objective << " || Train cost: " << train_cost << " || Test cost: " << test_cost << " || Train acc: " << train_accuracy * 100.0f << " % || Test acc: " << test_accuracy * 100.0f << " % || Time: " << time.count() << " sec\n";
            }

            if (epoch == epochs)
                break;

            Forward cnn_frd_cache = conv_frdpass(model.params, Xtrain, model.cv_stats, model.mx_stats, model.activ, dropout);
            Backward cnn_grads = conv_bckprop(cnn_frd_cache, model.params, ytrain, model.cv_stats, model.mx_stats, model.activ, dropout, lambda_l1, lambda_l2);
            adam(model.params, cnn_grads, model.state, lr, beta1, beta2, eps);
        }

        if ((restore_best != 0 || restore_meaningful != 0) && snapshot != 0)
            model.params = move(best_params);

        return model;
    }

    Parameters clone_params(const Parameters& params)
    {
        Parameters newparams(params.W.size());

        for (int i = 1; i < params.W.size(); i++)
        {
            newparams.W[i] = params.W[i].clone();
            newparams.B[i] = params.B[i].clone();
        }

        return newparams;
    }

    CNN_model& change_params(CNN_model& model)
    {
        Tensor Replace = Tensor::fill(model.params.W[1].getShape(), 1.0f);

        model.params.W[1] = Replace.clone();

        return model;
    }

    CNN_model& change_params2(CNN_model& model)
    {
        Parameters best_params(model.params.W.size());

        best_params = clone_params(model.params);

        CPUTensor temp = best_params.W[1].toCPU();

        temp(7) = 7.7f;

        best_params.W[1] = temp.toCUDA().clone();

        model.params = clone_params(best_params);

        return model;
    }

    void predict(const CPUTensor& X_test, const CPUTensor& y_test, const Parameters& params, const string& activation, int imgIdx)
    {
        CPUTensor one_mnist({ X_test.getShape()[X_test.dim() - 2], 1 });

        for (int i = 0; i < X_test.getShape()[X_test.dim() - 2]; i++)
        {
            one_mnist(i) = X_test(i * X_test.getShape()[X_test.dim() - 1] + imgIdx);
        }

        CPUTensor one_mnisty({ y_test.getShape()[y_test.dim() - 2], 1 });

        for (int i = 0; i < y_test.getShape()[y_test.dim() - 2]; i++)
        {
            one_mnisty(i) = y_test(i * y_test.getShape()[y_test.dim() - 1] + imgIdx);
        }

        int L = static_cast<int>(params.W.size());

        int pred_label = 0;

        if (y_test.getShape()[y_test.dim() - 2] > 1)
        {
            Forward frd_cache1 = forward_pass(params, one_mnist.toCUDA(), activation);

            pred_label = Tensor::argmax(frd_cache1.A[L - 1]).toScalar();

            int truth_label = Tensor::argmax(one_mnisty.toCUDA()).toScalar();

            cout << "\nTruth: " << truth_label << " || Pred: " << pred_label;

            printMnistImage(X_test, imgIdx);
        }
        else
        {
            Forward frd_cache1 = forward_pass(params, one_mnist.toCUDA(), activation);

            pred_label = (frd_cache1.A[L - 1] > 0.5f).toCPU()(0);

            cout << "\nTruth: " << y_test(imgIdx) << " || Pred: " << pred_label;
        }
    }

    void predict_cnn(const CNN_model& model, const Tensor& X, const Tensor& y, int idx)
    {
        CPUTensor one_img = choose_img(X.toCPU(), idx);

        float label = choose_label(y, idx);

        Forward frd_cache = conv_frdpass(model.params, one_img.toCUDA(), model.cv_stats, model.mx_stats, model.activ);

        int L = static_cast<int>(model.params.W.size());

        float prediction = 0.0f;

        if (y.getShape()[0] == 1)
        {
            prediction = (frd_cache.A[L - 1] > 0.5f).toScalar();

            string lbl = label == 0.0f ? "no-meteor" : "meteor";
            string prd = prediction == 0.0f ? "no-meteor" : "meteor";

            cout << "\nTruth: " << lbl << " ||| Pred: " << prd;
        }
        else
        {
            prediction = Tensor::argmax(frd_cache.A[L - 1], 0).toScalar();

            cout << "\nTruth: " << label << " ||| Pred: " << prediction;
        }
    }
}