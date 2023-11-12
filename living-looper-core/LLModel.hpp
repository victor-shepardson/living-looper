#pragma once

#include "Resampler.hpp"
#include <torch/script.h>

namespace LivingLooper {

// typedef torch::TensorAccessor<float, 3> ll_audio_t;
// typedef torch::TensorAccessor<float, 2> ll_latent_t;
// typedef torch::Tensor ll_audio_t;
// typedef torch::Tensor ll_latent_t;

// typedef tuple<ll_audio_t, ll_latent_t> ll_return_t;
// typedef torch::IValue ll_return_t;

#define MAX_LOOPS 32
#define SWAP(A, B) auto temp = A; A = B; B = temp;
#define RANGE(I, N) for(int I=0; I<N; I++)
#define PRINT(X) std::cout << X << std::endl;
#define TORCH_GUARD ;
// #define TORCH_GUARD c10::InferenceMode guard;


// LLModel encapsulates the libtorch parts
struct LLModel {

    int loop_idx, thru, auto_mode;

    torch::jit::Module model;

    int host_sr, sr;
    int block_size;
    // int z_per_second;
    int n_latent;
    int n_loops;
    std::atomic<bool> loaded;

    std::unique_ptr<std::thread> load_thread;
    std::unique_ptr<std::thread> compute_thread;

    std::vector<torch::jit::IValue> model_args;
    std::vector<torch::jit::IValue> model_args_empty;
    torch::ivalue::TupleElements model_outs;

    float delay; // estimated total delay in seconds
    std::unique_ptr<Resampler> res_in;
    std::vector<Resampler> res_out;
    long m_internal_samples; // count of total samples processed
    int m_processing_latency; // in model samples

    torch::Tensor write_buffer;
    torch::Tensor read_buffer;
    torch::Tensor latent_buffer;
    size_t write_idx, read_idx;
    int latent_idx;

    LLModel(int host_sr) : loaded(false) {
        this->host_sr = host_sr;

        loop_idx = 0;
        thru = 0;

        // TODO: expose processing latency to user
        // defaults to block_size - 1
        m_processing_latency = -1;
        // count of model samples elapsed
        m_internal_samples = 0;

        read_idx = 0;
        write_idx = 0;
        latent_idx = -1;

        // at::init_num_threads();
        // at::set_num_threads(1);
        // torch::set_num_threads(1);

        // unsigned int num_threads = std::thread::hardware_concurrency();
        torch::jit::getProfilingMode() = false;
        TORCH_GUARD;
        torch::jit::setGraphExecutorOptimize(true);
    }

    ~LLModel() {
        TORCH_GUARD;
        if (load_thread && load_thread->joinable()) {
            load_thread->join();
        }
        if (compute_thread && compute_thread->joinable()) {
            compute_thread->join();
        }
    }

    void reset() {
        TORCH_GUARD;
        model.get_method("reset")(model_args_empty);
    }

    void forward() {
        TORCH_GUARD;
        model_outs = model(model_args).toTupleRef().elements();
    }

    // ingest a single sample of audio input
    void write(float x) {
        TORCH_GUARD;

        // write to resampler
        res_in->write(x);

        // while there are new resampled values, write them into buffer
        while(res_in->pending()){
            // std::cout << "read from res_in" << std::endl;
            write_buffer[0][0][write_idx] = res_in->read();
            write_idx += 1;
            m_internal_samples += 1;
            if (write_idx == block_size){
                dispatch();
                write_idx = 0;
            }
        }
    } 
    // return a single sample of audio output
    void read(float * buf, float * latent_buf) {
        TORCH_GUARD;
        float x[MAX_LOOPS];
        RANGE(i, n_loops) x[i] = 0;

        // process model samples until an output sample is ready
        while (!res_out[0].pending()){
            // write zeros until first buffer is full,
            // plus processing time has elapsed
            if (m_internal_samples >= block_size + m_processing_latency){
                if (read_idx % block_size == 0){
                    join();
                    read_idx = 0;
                }
                auto acc = read_buffer.accessor<float, 3>();
                RANGE(i, n_loops) x[i] = acc[i][0][read_idx];
                read_idx += 1;
            }    
            // std::cout << "write to res_out" << std::endl;
            RANGE(i, n_loops) res_out[i].write(x[i]);
        }
        // std::cout << "read from res_out" << std::endl;
        RANGE(i, n_loops) buf[i] = res_out[i].read();

        // std::cout << "DIM " << latent_buffer.sizes() << std::endl;
        if((latent_idx >= 0) && (latent_idx < latent_buffer.size(2))){
            // std::cout << "IDX" << latent_idx << std::endl;
            auto acc = latent_buffer.accessor<float, 3>();
            RANGE(i, n_loops) latent_buf[i] = acc[i][0][latent_idx];
            // RANGE(i, n_loops) PRINT(i << " " << latent_idx << " " << latent_buf[i]);
            latent_idx++;
        } else {
            RANGE(i, n_loops) latent_buf[i] = 0.;
        }
    }
    // read and write should be used together like so:
    void step(float from, float * to, float * latent){
        write(from); read(to, latent);
    }

    // start the next block of processing
    void dispatch() {
        TORCH_GUARD;
        // PRINT("dispatch");

        if (compute_thread && compute_thread->joinable()) 
            PRINT("ERROR: trying to start compute_thread before previous one is finished");

        // swap buffers
        auto temp = write_buffer;
        write_buffer = model_args[1].toTensor();
        model_args[1] = temp;

        model_args[0] = torch::IValue(loop_idx); 
        model_args[2] = torch::IValue(thru); 
        model_args[3] = torch::IValue(auto_mode); 

        compute_thread = std::make_unique<std::thread>(&LLModel::forward, this);
        // PRINT("dispatch complete");
    } 
    // finish the last block of processing
    void join() {
        TORCH_GUARD;
        // join model thread
        // PRINT("join");
        if (!compute_thread) 
            PRINT("ERROR: no compute thread");
        if (!compute_thread->joinable()) 
            PRINT("ERROR: compute thread not joinable");

        compute_thread->join();

        read_buffer = model_outs[0].toTensor().clone();
        latent_buffer = model_outs[1].toTensor().clone();
        latent_idx = 0;
    }
    
    void _load(const std::string& filename) {
        TORCH_GUARD;
        try {
            model = torch::jit::load(filename);
            model.eval();
            // this->model = torch::jit::optimize_for_inference(this->model);
        }
        catch (const c10::Error& e) {
        // why no error when filename is bad?
            PRINT(e.what());
            PRINT(e.msg());
            PRINT("error loading the model\n");
            return;
        }

        this->block_size = this->n_latent = this->sr = this->n_loops = -1;

        for (auto const& attr: model.named_attributes()) {
            if (attr.name == "n_loops") {n_loops = attr.value.toInt();} 
            if (attr.name == "block_size") {block_size = attr.value.toInt();} 
            if (attr.name == "sampling_rate") {sr = attr.value.toInt();} 
            if (attr.name == "n_latent") {n_latent = attr.value.toInt();} 
        }
        // this->z_per_second = this->sr / this->block_size;

        if ((block_size<=0) || 
            (n_loops<=0) || 
            (n_latent<0) || 
            (sr<=0)){
        PRINT("model load failed");
        return;
        }

        PRINT("\tnumber of loops: " << n_loops );
        PRINT("\tblock size: " << block_size );
        PRINT("\tsample rate: " << sr );
        PRINT("\tlatent size: " << n_latent );

        // TORCH_GUARD;
        model_args.clear();
        model_args.push_back(torch::IValue(0)); //loop
        model_args.push_back(torch::ones({1,1,block_size})); //audio
        model_args.push_back(torch::IValue(0)); //thru
        model_args.push_back(torch::IValue(0)); //auto

        delay = (block_size + m_processing_latency)/sr;
        res_in = std::make_unique<Resampler>(host_sr, sr, 3);
        delay += res_in->delay;
        for (int i=0; i<n_loops; i++){
            res_out.push_back(Resampler(sr, host_sr, 3));
        }
        delay += res_out[0].delay;

        write_buffer = torch::zeros({1,1,block_size});

        if (m_processing_latency < 0){
            m_processing_latency = block_size - 1;
        }

        loaded = true;
    }
    void load(const std::string& filename) {
        load_thread = std::make_unique<std::thread>(
            &LLModel::_load, this, filename);
    }

    // void forward(float* input, int loop_idx, int oneshot, float* outBuffer) {
    //     c10::InferenceMode guard;

    //     auto ivalue = forwardIValue(input, loop_idx, oneshot);
    //     // auto acc = get<0>(accs);
    //     auto tup = model(model_args).toTupleRef();
    //     auto acc = tup.elements()[0].toTensor().accessor<float,3>();
    //     // auto acc = get<0>(this->model(model_args).toTuple()).toTensor().accessor<float,3>();

    //     for(int j=0; j<acc.size(0); j++) {
    //         for(int i=0; i<acc.size(2); i++) {
    //             outBuffer[n_loops*i + j] = acc[j][0][i];
    //         }
    //     }
    // }
};

} //namespace LivingLooper