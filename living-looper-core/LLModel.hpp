#pragma once

#include <torch/script.h>

using namespace std;

namespace LivingLooper {

// typedef torch::TensorAccessor<float, 3> ll_audio_t;
// typedef torch::TensorAccessor<float, 2> ll_latent_t;
typedef torch::Tensor ll_audio_t;
typedef torch::Tensor ll_latent_t;

// typedef tuple<ll_audio_t, ll_latent_t> ll_return_t;
typedef torch::IValue ll_return_t;

// LLModel encapsulates the libtorch parts
struct LLModel {

    torch::jit::Module model;

    int sr;
    int block_size;
    int z_per_second;
    int latent_size;
    int n_loops;
    bool loaded;

    vector<torch::jit::IValue> inputs_rave;
    vector<torch::jit::IValue> inputs_empty;

LLModel() {
    // at::init_num_threads();
    // at::set_num_threads(1);
    // torch::set_num_threads(1);

//    unsigned int num_threads = std::thread::hardware_concurrency();
    this->loaded=false;
    torch::jit::getProfilingMode() = false;
    c10::InferenceMode guard;
    torch::jit::setGraphExecutorOptimize(true);
    }
    
  void load(const std::string& rave_model_file) {
    // std::cout << "\"" <<rave_model_file << "\"" <<std::endl;
    try {
        c10::InferenceMode guard;
        this->model = torch::jit::load(rave_model_file);
        this->model.eval();
        // this->model = torch::jit::optimize_for_inference(this->model);
    }
    catch (const c10::Error& e) {
      // why no error when filename is bad?
        cout << e.what();
        cout << e.msg();
        cout << "error loading the model\n";
        return;
    }

    // support for Neutone models
    // if (this->model.hasattr("model")){
    //   this->model = this->model.attr("model").toModule();
    // }

    this->block_size = this->latent_size = this->sr = this->n_loops = -1;

    for (auto const& attr: model.named_attributes()) {
        if (attr.name == "n_loops") {this->n_loops = attr.value.toInt();} 
        if (attr.name == "block_size") {this->block_size = attr.value.toInt();} 
        if (attr.name == "sampling_rate") {this->sr = attr.value.toInt();} 
    }
    // this->z_per_second = this->sr / this->block_size;

    if ((this->block_size<=0) || 
        (this->n_loops<=0) || 
        // (this->latent_size<0) || 
        (this->sr<=0)){
      cout << "model load failed" << std::endl;
      return;
    }

    cout << "\tnumber of loops: " << this->n_loops << endl;
    cout << "\tblock size: " << this->block_size << endl;
    // std::cout << "\tlatent size: " << this->latent_size << std::endl;
    cout << "\tsample rate: " << this->sr << endl;

    c10::InferenceMode guard;
    inputs_rave.clear();
    inputs_rave.push_back(torch::IValue(0));
    inputs_rave.push_back(torch::ones({1,1,block_size}));
    inputs_rave.push_back(torch::IValue(0));

    this->loaded = true;
  }

  void reset () {
    c10::InferenceMode guard;
    this->model.get_method("reset")(inputs_empty);
  }

  ll_return_t forwardIValue(
      float* input, int loop_idx, int oneshot) {
    c10::InferenceMode guard;

    inputs_rave[0] = torch::IValue(loop_idx); 

    inputs_rave[1] = torch::from_blob(
      input, block_size).reshape({1, 1, block_size});

    inputs_rave[2] = torch::IValue(oneshot); 

    return this->model(inputs_rave);
  }

  void forward(float* input, int loop_idx, int oneshot, float* outBuffer) {
    c10::InferenceMode guard;

    auto ivalue = forwardIValue(input, loop_idx, oneshot);
    // auto acc = get<0>(accs);
    auto tup = this->model(inputs_rave).toTupleRef();
    auto acc = tup.elements()[0].toTensor().accessor<float,3>();
    // auto acc = get<0>(this->model(inputs_rave).toTuple()).toTensor().accessor<float,3>();

    for(int j=0; j<acc.size(0); j++) {
      for(int i=0; i<acc.size(2); i++) {
        outBuffer[n_loops*i + j] = acc[j][0][i];
      }
    }
  }

};

} //namespace LivingLooper