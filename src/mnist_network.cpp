#include "mnist_network.h"
#include <fstream>

mnist_network::mnist_network(std::vector<int> layers) {
  LAYER_SIZES = layers;
  NET_SIZE = layers.size();
  biases = static_network_utils::create_rand_2d_arr(layers.size(), layers[0], -1, 1);
  errors = static_network_utils::create_rand_2d_arr(layers.size(), layers[0], -1, 1);
  outputs = static_network_utils::create_rand_2d_arr(layers.size(), layers[0], -1, 1);
  weights = static_network_utils::init_weights(layers, -1, 1);
}

std::vector<double> mnist_network::calcOutputs(std::vector<double> inVec) {
  outputs[0] = inVec;
  for (int layer = 1; layer < NET_SIZE; ++layer) {
    for (int neuron = 0; neuron < LAYER_SIZES[layer]; ++neuron) {
      double sum = biases[layer][neuron];
      for (int prev_neuron = 0; prev_neuron < LAYER_SIZES[layer - 1]; ++prev_neuron) {
        sum += weights[layer - 1][prev_neuron][neuron] * outputs[layer - 1][prev_neuron];
      }
      outputs[layer][neuron] = static_network_utils::sigmoid(sum);
    }
  }
  return outputs[NET_SIZE - 1];
}

void mnist_network::calcErrors(std::vector<int> targetVec) {
  for (int neuron = 0; neuron < LAYER_SIZES[NET_SIZE - 1]; ++neuron) {
    double output = outputs[NET_SIZE - 1][neuron];
    errors[NET_SIZE - 1][neuron] = (output - targetVec[neuron]) * output * (1 - output);
  }
  for (int layer = NET_SIZE - 2; layer > 0; --layer) {
    for (int neuron = 0; neuron < LAYER_SIZES[layer]; ++neuron) {
      double sum = 0;
      for (int next_neuron = 0; next_neuron < LAYER_SIZES[layer + 1]; ++next_neuron) {
        sum += weights[layer][neuron][next_neuron] * errors[layer + 1][next_neuron];
      }
      errors[layer][neuron] = sum * outputs[layer][neuron] * (1 - outputs[layer][neuron]);
    }
  }
}

void mnist_network::updateNetwork(double learn_rate) {
  for (int layer = 1; layer < NET_SIZE; ++layer) {
    for (int neuron = 0; neuron < LAYER_SIZES[layer]; ++neuron) {
      biases[layer][neuron] -= learn_rate * errors[layer][neuron];
      for (int prev_neuron = 0; prev_neuron < LAYER_SIZES[layer - 1]; ++prev_neuron) {
        weights[layer - 1][prev_neuron][neuron] -= learn_rate * errors[layer][neuron] * outputs[layer - 1][prev_neuron];
      }
    }
  }
}

void mnist_network::save_to_file(std::string filename) {

  std::ofstream out;
  out.open(filename);

  out << "netsize\n." << NET_SIZE;

  out << "\nlayers\n";

  for (auto it = LAYER_SIZES.begin(); it != LAYER_SIZES.end(); ++it) {
    out << "." << *it;
  }

  out << "\nbiases\n";

  for (int rows = 0; rows < biases.size(); ++rows) {
    for (int cols = 0; cols < biases[0].size(); ++cols) {
      out << biases[rows][cols] << " ";
    }
    out << "\n";
  }

  out << "\nerrors\n";

  for (int rows = 0; rows < errors.size(); ++rows) {
    for (int cols = 0; cols < errors[0].size(); ++cols) {
      out << errors[rows][cols] << " ";
    }
    out << "\n";
  }

  out << "\noutputs\n";

  for (int rows = 0; rows < outputs.size(); ++rows) {
    for (int cols = 0; cols < outputs[0].size(); ++cols) {
      out << outputs[rows][cols] << " ";
    }
    out << "\n";
  }

  int weights_index = 0;

  for (int layer = 0; layer < weights.size() - 1; ++layer) {
    out << "weights" << weights_index << std::endl;
    ++weights_index;
    for (int row = 0; row < weights[layer].size(); ++row) {
      for (int col = 0; col < weights[layer][row].size(); ++col) {
        out << weights[layer][row][col] << " ";
      }
      out << "\n";
    }
  }

  out.close();
}
std::ostream &operator<<(std::ostream &out, const mnist_network &net) {
  out << "Network size: " << net.NET_SIZE << std::endl;
  return out;
}
void mnist_network::train(mnist_set ms, double learn_rate) {
  for (image i : ms.set) {
    calcOutputs(i.data);
    std::vector<int> target(10);
    target[i.label] = 1;
    calcErrors(target);
    updateNetwork(learn_rate);
  }
}
void mnist_network::test(mnist_set ms) {
  int correct = 0;
  for (image i : ms.set) {
    if (static_network_utils::max_element(calcOutputs(i.data)) == i.label) ++correct;
    //std::cout<<"\n"<<i.label<<", net guess: " <<static_network_utils::max_element(calcOutputs(i.data)) <<std::endl;
    for (double d : calcOutputs(i.data)) {std::cout<<d<<" ";}
    std::cout<<calcOutputs(i.data).size();
    std::cout<<std::endl;
  }
  double percent = static_cast<double>(correct) / ms.set.size();
  std::cout<< percent*100 <<"% correct";
}
