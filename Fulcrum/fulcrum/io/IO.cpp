#include <fulcrum/io/IO.h>

namespace fulcrum::io
{
    void save_vector_bin(const vector<float>& x, const filesystem::path& path)
    {
        ofstream outbin(path, ios::binary | ios::trunc);

        if (!outbin)
            throw runtime_error("Unable to open write file!");

        outbin.write(reinterpret_cast<const char*>(x.data()), x.size() * sizeof(float));

        if (!outbin)
            throw runtime_error("Unable to write to file!");
    }

    void load_vector_bin(vector<float>& x, const filesystem::path& path)
    {
        ifstream inbin(path, ios::binary);

        if (!inbin)
            throw runtime_error("Unable to open read file!");

        if (!(inbin.read(reinterpret_cast<char*>(x.data()), x.size() * sizeof(float))))
            throw runtime_error("Unable to read the file!");
    }

    void save_tensor(const Tensor& T, const filesystem::path& path)
    {
        ofstream tout(path, ios::binary | ios::trunc);

        if (!tout)
            throw runtime_error("Unable to open write file!");

        const char magic[4] = { 'F', 'L', 'C', 'T' };

        float v = 0.01f;

        float dtype = T.getDataType() == DataType::float32 ? 0.0f : 1.0f;

        int rank = T.dim();

        tout.write(magic, sizeof(magic));

        tout.write(reinterpret_cast<const char*>(&v), sizeof(v));

        tout.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));

        tout.write(reinterpret_cast<const char*>(&rank), sizeof(rank));

        tout.write(reinterpret_cast<const char*>(T.getShape().data()), rank * sizeof(int));

        if (!dtype)
            tout.write(reinterpret_cast<const char*>(T.toCPU().getFloatData()), T.size() * sizeof(float));
        else
            tout.write(reinterpret_cast<const char*>(T.toCPU().getIntData()), T.size() * sizeof(int));
        
        if (!tout)
            throw runtime_error("Unable to finish file write!");
    }

    Tensor load_tensor(const filesystem::path& path)
    {
        ifstream tin(path, ios::binary);

        if (!tin)
            throw runtime_error("Unable to open read file!");

        const char magic[4] = { 'F', 'L', 'C', 'T' };

        char load_magic[4] = { 0 };

        if (!(tin.read(load_magic, sizeof(load_magic))))
            throw runtime_error("Unable to read magic number!");

        if (memcmp(load_magic, magic, 4) != 0)
            throw runtime_error("Invalid file format!");

        float v = 0.01f;

        float load_v = 0.0f;

        if (!(tin.read(reinterpret_cast<char*>(&load_v), sizeof(load_v))))
            throw runtime_error("Unable to read version number!");

        if (load_v != v)
            throw runtime_error("Invalid file version!");

        float load_dtype = 0.0f;

        if (!(tin.read(reinterpret_cast<char*>(&load_dtype), sizeof(load_dtype))))
            throw runtime_error("Unable to read data type!");

        DataType tdtype = load_dtype == 0.0f ? DataType::float32 : DataType::int32;

        int load_rank = 0;

        if (!(tin.read(reinterpret_cast<char*>(&load_rank), sizeof(load_rank))))
            throw runtime_error("Unable to read rank!");

        vector<int> load_shape(load_rank);

        if (!(tin.read(reinterpret_cast<char*>(load_shape.data()), load_rank * sizeof(int))))
            throw runtime_error("Unable to read tensor shape!");

        Tensor res(load_shape, tdtype);

        if (tdtype == DataType::float32)
        {
            vector<float> payload(res.size());

            if (!(tin.read(reinterpret_cast<char*>(payload.data()), res.size() * sizeof(float))))
                throw runtime_error("Unable to read tensor payload!");

            res = payload;
        }
        else
        {
            vector<int> payload(res.size());

            if (!(tin.read(reinterpret_cast<char*>(payload.data()), res.size() * sizeof(int))))
                throw runtime_error("Unable to read tensor payload!");

            res = payload;
        }
        
        return res;
    }

    void save_model(const Parameters& params, const filesystem::path& path)
    {
        ofstream pout(path, ios::binary | ios::trunc);

        if (!pout)
            throw runtime_error("Unable to open write file!");

        const char magic[4] = { 'F', 'L', 'C', 'T' };
        float v = 0.01f;
        int dtype = params.W[1].getDataType() == DataType::float32 ? 0 : 1;
        int dim = params.W.size();

        pout.write(magic, sizeof(magic));

        pout.write(reinterpret_cast<const char*>(&v), sizeof(v));

        pout.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));

        pout.write(reinterpret_cast<const char*>(&dim), sizeof(dim));

        for (int i = 1; i < dim; i++)
        {
            int rankW = params.W[i].dim();
            int rankB = params.B[i].dim();

            pout.write(reinterpret_cast<const char*>(&rankW), sizeof(rankW));
            pout.write(reinterpret_cast<const char*>(&rankB), sizeof(rankB));

            pout.write(reinterpret_cast<const char*>(params.W[i].getShape().data()), rankW * sizeof(int));
            pout.write(reinterpret_cast<const char*>(params.B[i].getShape().data()), rankB * sizeof(int));

            if (dtype == 0)
            {
                pout.write(reinterpret_cast<const char*>(params.W[i].toCPU().getFloatData()), params.W[i].byteSize());
                pout.write(reinterpret_cast<const char*>(params.B[i].toCPU().getFloatData()), params.B[i].byteSize());
            }
            else
            {
                pout.write(reinterpret_cast<const char*>(params.W[i].toCPU().getIntData()), params.W[i].byteSize());
                pout.write(reinterpret_cast<const char*>(params.B[i].toCPU().getIntData()), params.B[i].byteSize());
            }
        }

        if (!pout)
            throw runtime_error("Uable to finish file write!");
    }

    Parameters load_model(const filesystem::path& path)
    {
        ifstream pin(path, ios::binary);

        if (!pin)
            throw runtime_error("Unable to open read file!");

        const char magic[4] = { 'F', 'L', 'C', 'T' };
        char load_magic[4] = { 0 };

        if (!(pin.read(load_magic, sizeof(load_magic))))
            throw runtime_error("Unable to read magic number!");

        if (memcmp(load_magic, magic, 4))
            throw runtime_error("Invalid file format!");

        float v = 0.01f;
        float load_v = 0.0f;

        if (!(pin.read(reinterpret_cast<char*>(&load_v), sizeof(load_v))))
            throw runtime_error("Unable to read version number!");

        if (load_v != v)
            throw runtime_error("Invalid file version!");

        int load_type = 0;

        if (!(pin.read(reinterpret_cast<char*>(&load_type), sizeof(load_type))))
            throw runtime_error("Unable to load data type!");

        DataType dtype = load_type == 0 ? DataType::float32 : DataType::int32;

        int load_dim = 0;

        if (!(pin.read(reinterpret_cast<char*>(&load_dim), sizeof(load_dim))))
            throw runtime_error("Unable to read total dimension!");

        Parameters params(load_dim);

        for (int i = 1; i < load_dim; i++)
        {
            int load_rankW = 0;
            int load_rankB = 0;

            if (!(pin.read(reinterpret_cast<char*>(&load_rankW), sizeof(load_rankW))))
                throw runtime_error("Unable to read W rank for Tensor " + to_string(i));

            if (!(pin.read(reinterpret_cast<char*>(&load_rankB), sizeof(load_rankB))))
                throw runtime_error("Unable to read B rank for Tensor " + to_string(i));

            vector<int> load_shapeW(load_rankW);
            vector<int> load_shapeB(load_rankB);

            if (!(pin.read(reinterpret_cast<char*>(load_shapeW.data()), load_rankW * sizeof(int))))
                throw runtime_error("Unable to read W shape for Tensor " + to_string(i));

            if (!(pin.read(reinterpret_cast<char*>(load_shapeB.data()), load_rankB * sizeof(int))))
                throw runtime_error("Unable to read B shape for Tensor " + to_string(i));

            Tensor load_W(load_shapeW, dtype);
            Tensor load_B(load_shapeB, dtype);

            if (dtype == DataType::float32)
            {
                vector<float> load_ploadW(load_W.size());
                vector<float> load_ploadB(load_B.size());

                if (!(pin.read(reinterpret_cast<char*>(load_ploadW.data()), load_W.size() * sizeof(float))))
                    throw runtime_error("Unable to read W payload for Tensor " + to_string(i));

                if (!(pin.read(reinterpret_cast<char*>(load_ploadB.data()), load_B.size() * sizeof(float))))
                    throw runtime_error("Unable to read B payload for Tensor" + to_string(i));

                load_W = load_ploadW;
                load_B = load_ploadB;
            }
            else
            {
                vector<int> load_ploadW(load_W.size());
                vector<int> load_ploadB(load_B.size());

                if (!(pin.read(reinterpret_cast<char*>(load_ploadW.data()), load_W.size() * sizeof(int))))
                    throw runtime_error("Unable to read W payload for Tensor " + to_string(i));

                if (!(pin.read(reinterpret_cast<char*>(load_ploadB.data()), load_B.size() * sizeof(int))))
                    throw runtime_error("Unable to read B payload for Tensor" + to_string(i));

                load_W = load_ploadW;
                load_B = load_ploadB;
            }

            params.W[i] = move(load_W);
            params.B[i] = move(load_B);
        }

        return params;
    }

    void save_cnn(const CNN_model& model, const filesystem::path& path, bool save_adam)
    {
        ofstream mout(path, ios::binary | ios::trunc);

        if (!mout)
            throw runtime_error("Unable to open write file!");

        const char magic[4] = { 'F', 'L', 'C', 'T' };
        float v = 0.01f;
        int dtype = model.params.W[1].getDataType() == DataType::float32 ? 0 : 1;
        int dim = model.params.W.size();

        mout.write(magic, sizeof(magic));

        mout.write(reinterpret_cast<const char*>(&v), sizeof(v));

        mout.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));

        mout.write(reinterpret_cast<const char*>(&dim), sizeof(dim));

        for (int i = 1; i < dim; i++)
        {
            int rankW = model.params.W[i].dim();
            int rankB = model.params.B[i].dim();

            mout.write(reinterpret_cast<const char*>(&rankW), sizeof(rankW));
            mout.write(reinterpret_cast<const char*>(&rankB), sizeof(rankB));

            mout.write(reinterpret_cast<const char*>(model.params.W[i].getShape().data()), rankW * sizeof(int));
            mout.write(reinterpret_cast<const char*>(model.params.B[i].getShape().data()), rankB * sizeof(int));

            if (dtype == 0)
            {
                mout.write(reinterpret_cast<const char*>(model.params.W[i].toCPU().getFloatData()), model.params.W[i].byteSize());
                mout.write(reinterpret_cast<const char*>(model.params.B[i].toCPU().getFloatData()), model.params.B[i].byteSize());
            }
            else
            {
                mout.write(reinterpret_cast<const char*>(model.params.W[i].toCPU().getIntData()), model.params.W[i].byteSize());
                mout.write(reinterpret_cast<const char*>(model.params.B[i].toCPU().getIntData()), model.params.B[i].byteSize());
            }
        }

        int nr_layers = model.cv_stats.size();

        mout.write(reinterpret_cast<const char*>(&nr_layers), sizeof(nr_layers));

        mout.write(reinterpret_cast<const char*>(model.cv_stats.data()), nr_layers * sizeof(convStats));

        mout.write(reinterpret_cast<const char*>(model.mx_stats.data()), nr_layers * sizeof(MaxPoolStats));

        uint32_t activ_length = model.activ.size();

        mout.write(reinterpret_cast<const char*>(&activ_length), sizeof(activ_length));

        mout.write(model.activ.data(), activ_length);

        if (save_adam)
        {
            bool has_adam = true;

            mout.write(reinterpret_cast<const char*>(&has_adam), sizeof(has_adam));

            for (int i = 1; i < dim; i++)
            {
                int rankvdw = model.state.VdW[i].dim();
                int rankvdb = model.state.VdB[i].dim();
                int ranksdw = model.state.SdW[i].dim();
                int ranksdb = model.state.SdB[i].dim();

                mout.write(reinterpret_cast<const char*>(&rankvdw), sizeof(rankvdw));
                mout.write(reinterpret_cast<const char*>(&rankvdb), sizeof(rankvdb));
                mout.write(reinterpret_cast<const char*>(&ranksdw), sizeof(ranksdw));
                mout.write(reinterpret_cast<const char*>(&ranksdb), sizeof(ranksdb));

                mout.write(reinterpret_cast<const char*>(model.state.VdW[i].getShape().data()), rankvdw * sizeof(int));
                mout.write(reinterpret_cast<const char*>(model.state.VdB[i].getShape().data()), rankvdb * sizeof(int));
                mout.write(reinterpret_cast<const char*>(model.state.SdB[i].getShape().data()), ranksdw * sizeof(int));
                mout.write(reinterpret_cast<const char*>(model.state.SdB[i].getShape().data()), ranksdb * sizeof(int));

                if (dtype == 0)
                {
                    mout.write(reinterpret_cast<const char*>(model.state.VdW[i].toCPU().getFloatData()), model.state.VdW[i].byteSize());
                    mout.write(reinterpret_cast<const char*>(model.state.VdB[i].toCPU().getFloatData()), model.state.VdB[i].byteSize());
                    mout.write(reinterpret_cast<const char*>(model.state.SdW[i].toCPU().getFloatData()), model.state.SdW[i].byteSize());
                    mout.write(reinterpret_cast<const char*>(model.state.SdB[i].toCPU().getFloatData()), model.state.SdB[i].byteSize());
                }
                else
                {
                    mout.write(reinterpret_cast<const char*>(model.state.VdW[i].toCPU().getIntData()), model.state.VdW[i].byteSize());
                    mout.write(reinterpret_cast<const char*>(model.state.VdB[i].toCPU().getIntData()), model.state.VdB[i].byteSize());
                    mout.write(reinterpret_cast<const char*>(model.state.SdW[i].toCPU().getIntData()), model.state.SdW[i].byteSize());
                    mout.write(reinterpret_cast<const char*>(model.state.SdB[i].toCPU().getIntData()), model.state.SdB[i].byteSize());
                }
            }

            int t = model.state.t;

            mout.write(reinterpret_cast<const char*>(&t), sizeof(t));
        }

        if (!mout)
            throw runtime_error("Unable to finish file write!");
    }

    CNN_model load_cnn(const filesystem::path& path, bool restore_adam)
    {
        ifstream min(path, ios::binary);

        if (!min)
            throw runtime_error("Unable to open read file!");

        const char magic[4] = { 'F', 'L', 'C', 'T' };
        char load_magic[4] = { 0 };

        if (!(min.read(load_magic, sizeof(load_magic))))
            throw runtime_error("Unable to read magic number!");

        if (memcmp(load_magic, magic, 4))
            throw runtime_error("Invalid file format!");

        float v = 0.01f;
        float load_v = 0.0f;

        if (!(min.read(reinterpret_cast<char*>(&load_v), sizeof(load_v))))
            throw runtime_error("Unable to read version number!");

        if (load_v != v)
            throw runtime_error("Invalid file version!");

        int load_type = 0;

        if (!(min.read(reinterpret_cast<char*>(&load_type), sizeof(load_type))))
            throw runtime_error("Unable to load data type!");

        DataType dtype = load_type == 0 ? DataType::float32 : DataType::int32;

        int load_dim = 0;

        if (!(min.read(reinterpret_cast<char*>(&load_dim), sizeof(load_dim))))
            throw runtime_error("Unable to read total dimension!");

        Parameters load_params(load_dim);

        for (int i = 1; i < load_dim; i++)
        {
            int load_rankW = 0;
            int load_rankB = 0;

            if (!(min.read(reinterpret_cast<char*>(&load_rankW), sizeof(load_rankW))))
                throw runtime_error("Unable to read W rank for Tensor " + to_string(i));

            if (!(min.read(reinterpret_cast<char*>(&load_rankB), sizeof(load_rankB))))
                throw runtime_error("Unable to read B rank for Tensor " + to_string(i));

            vector<int> load_shapeW(load_rankW);
            vector<int> load_shapeB(load_rankB);

            if (!(min.read(reinterpret_cast<char*>(load_shapeW.data()), load_rankW * sizeof(int))))
                throw runtime_error("Unable to read W shape for Tensor " + to_string(i));

            if (!(min.read(reinterpret_cast<char*>(load_shapeB.data()), load_rankB * sizeof(int))))
                throw runtime_error("Unable to read B shape for Tensor " + to_string(i));

            Tensor load_W(load_shapeW, dtype);
            Tensor load_B(load_shapeB, dtype);

            if (dtype == DataType::float32)
            {
                vector<float> load_ploadW(load_W.size());
                vector<float> load_ploadB(load_B.size());

                if (!(min.read(reinterpret_cast<char*>(load_ploadW.data()), load_W.size() * sizeof(float))))
                    throw runtime_error("Unable to read W payload for Tensor " + to_string(i));

                if (!(min.read(reinterpret_cast<char*>(load_ploadB.data()), load_B.size() * sizeof(float))))
                    throw runtime_error("Unable to read B payload for Tensor" + to_string(i));

                load_W = load_ploadW;
                load_B = load_ploadB;
            }
            else
            {
                vector<int> load_ploadW(load_W.size());
                vector<int> load_ploadB(load_B.size());

                if (!(min.read(reinterpret_cast<char*>(load_ploadW.data()), load_W.size() * sizeof(int))))
                    throw runtime_error("Unable to read W payload for Tensor " + to_string(i));

                if (!(min.read(reinterpret_cast<char*>(load_ploadB.data()), load_B.size() * sizeof(int))))
                    throw runtime_error("Unable to read B payload for Tensor" + to_string(i));

                load_W = load_ploadW;
                load_B = load_ploadB;
            }

            load_params.W[i] = move(load_W);
            load_params.B[i] = move(load_B);
        }

        int load_layer_nr = 0;

        if (!(min.read(reinterpret_cast<char*>(&load_layer_nr), sizeof(load_layer_nr))))
            throw runtime_error("Unable to read number of layers!");

        vector<convStats> load_cv(load_layer_nr);

        if (!(min.read(reinterpret_cast<char*>(load_cv.data()), load_layer_nr * sizeof(convStats))))
            throw runtime_error("Unable to read conv stats!");

        vector<MaxPoolStats> load_mx(load_layer_nr);

        if (!(min.read(reinterpret_cast<char*>(load_mx.data()), load_layer_nr * sizeof(MaxPoolStats))))
            throw runtime_error("Unbale to read max pool stats!");

        uint32_t activ_length = 0;

        if (!(min.read(reinterpret_cast<char*>(&activ_length), sizeof(activ_length))))
            throw runtime_error("Unable to read activation string length!");

        string load_activ(activ_length, '\0');

        if (!(min.read(load_activ.data(), activ_length)))
            throw runtime_error("Unable to read activation string!");

        if (restore_adam)
        {
            bool load_has_adam = false;

            if (!(min.read(reinterpret_cast<char*>(&load_has_adam), sizeof(load_has_adam))))
                throw runtime_error("Unable to read adam confirmation variable!");

            if (load_has_adam)
            {
                AdamState load_state(load_params);

                for (int i = 1; i < load_dim; i++)
                {
                    int load_rankvdw = 0;
                    int load_rankvdb = 0;
                    int load_ranksdw = 0;
                    int load_ranksdb = 0;

                    if (!(min.read(reinterpret_cast<char*>(&load_rankvdw), sizeof(load_rankvdw))))
                        throw runtime_error("Unable to read the rank for VdW Tensor " + to_string(i));

                    if (!(min.read(reinterpret_cast<char*>(&load_rankvdb), sizeof(load_rankvdb))))
                        throw runtime_error("Unable to read the rank for VdB Tensor " + to_string(i));

                    if (!(min.read(reinterpret_cast<char*>(&load_ranksdw), sizeof(load_ranksdw))))
                        throw runtime_error("Unable to read the rank for SdW Tensor " + to_string(i));

                    if (!(min.read(reinterpret_cast<char*>(&load_ranksdb), sizeof(load_ranksdb))))
                        throw runtime_error("Unable to read the rank for SdB Tensor " + to_string(i));

                    vector<int> load_shapevdw(load_rankvdw);
                    vector<int> load_shapevdb(load_rankvdb);
                    vector<int> load_shapesdw(load_ranksdw);
                    vector<int> load_shapesdb(load_ranksdb);

                    if (!(min.read(reinterpret_cast<char*>(load_shapevdw.data()), load_rankvdw * sizeof(int))))
                        throw runtime_error("Unable to read the shape for VdW Tensor " + to_string(i));

                    if (!(min.read(reinterpret_cast<char*>(load_shapevdb.data()), load_rankvdb * sizeof(int))))
                        throw runtime_error("Unable to read the shape for VdB Tensor " + to_string(i));

                    if (!(min.read(reinterpret_cast<char*>(load_shapesdw.data()), load_ranksdw * sizeof(int))))
                        throw runtime_error("Unable to read the shape for SdW Tensor " + to_string(i));

                    if (!(min.read(reinterpret_cast<char*>(load_shapesdb.data()), load_ranksdb * sizeof(int))))
                        throw runtime_error("Unable to read the shape for SdB Tensor " + to_string(i));

                    Tensor loadvdw(load_shapevdw, dtype);
                    Tensor loadvdb(load_shapevdb, dtype);
                    Tensor loadsdw(load_shapesdw, dtype);
                    Tensor loadsdb(load_shapesdb, dtype);

                    if (dtype == DataType::float32)
                    {
                        vector<float> load_ploadvdw(loadvdw.size());
                        vector<float> load_ploadvdb(loadvdb.size());
                        vector<float> load_ploadsdw(loadsdw.size());
                        vector<float> load_ploadsdb(loadsdb.size());

                        if (!(min.read(reinterpret_cast<char*>(load_ploadvdw.data()), loadvdw.size() * sizeof(float))))
                            throw runtime_error("Unable to read the payload for VdW Tensor " + to_string(i));

                        if (!(min.read(reinterpret_cast<char*>(load_ploadvdb.data()), loadvdb.size() * sizeof(float))))
                            throw runtime_error("Unable to read the payload for VdB Tensor" + to_string(i));

                        if (!(min.read(reinterpret_cast<char*>(load_ploadsdw.data()), loadsdw.size() * sizeof(float))))
                            throw runtime_error("Unable to read the payload for SdW Tensor " + to_string(i));

                        if (!(min.read(reinterpret_cast<char*>(load_ploadsdb.data()), loadsdb.size() * sizeof(float))))
                            throw runtime_error("Unable to read the payload for SdB Tensor " + to_string(i));

                        loadvdw = load_ploadvdw;
                        loadvdb = load_ploadvdb;
                        loadsdw = load_ploadsdw;
                        loadsdb = load_ploadsdb;
                    }
                    else
                    {
                        vector<int> load_ploadvdw(loadvdw.size());
                        vector<int> load_ploadvdb(loadvdb.size());
                        vector<int> load_ploadsdw(loadsdw.size());
                        vector<int> load_ploadsdb(loadsdb.size());

                        if (!(min.read(reinterpret_cast<char*>(load_ploadvdw.data()), loadvdw.size() * sizeof(int))))
                            throw runtime_error("Unable to read the payload for VdW Tensor " + to_string(i));

                        if (!(min.read(reinterpret_cast<char*>(load_ploadvdb.data()), loadvdb.size() * sizeof(int))))
                            throw runtime_error("Unable to read the payload for VdB Tensor" + to_string(i));

                        if (!(min.read(reinterpret_cast<char*>(load_ploadsdw.data()), loadsdw.size() * sizeof(int))))
                            throw runtime_error("Unable to read the payload for SdW Tensor " + to_string(i));

                        if (!(min.read(reinterpret_cast<char*>(load_ploadsdb.data()), loadsdb.size() * sizeof(int))))
                            throw runtime_error("Unable to read the payload for SdB Tensor " + to_string(i));

                        loadvdw = load_ploadvdw;
                        loadvdb = load_ploadvdb;
                        loadsdw = load_ploadsdw;
                        loadsdb = load_ploadsdb;
                    }

                    load_state.VdW[i] = move(loadvdw);
                    load_state.VdB[i] = move(loadvdb);
                    load_state.SdW[i] = move(loadsdw);
                    load_state.SdB[i] = move(loadsdb);
                }

                int load_t = 0;

                if (!(min.read(reinterpret_cast<char*>(&load_t), sizeof(int))))
                    throw runtime_error("Unable to load adam t variable!");

                load_state.t = load_t;

                CNN_model load_model(load_params, load_cv, load_mx, load_activ, load_state);

                return load_model;
            }
            else
                throw runtime_error("The file does not have a saved adam state!");
        }

        CNN_model load_model(load_params, load_cv, load_mx, load_activ);

        return load_model;
    }
}