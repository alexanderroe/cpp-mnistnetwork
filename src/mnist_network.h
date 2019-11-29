#ifndef MNISTNETWORK__MNIST_NETWORK_H_
#define MNISTNETWORK__MNIST_NETWORK_H_
#include <iostream>
#include <vector>
#include <cmath>
#include "mnist_set.h"

class mnist_network {
  // constant
  static constexpr size_t IMAGE_SIZE = 28;

  // class variables
  std::vector<int> LAYER_SIZES;
  int NET_SIZE;
  std::vector<std::vector<double>> biases, errors, outputs;
  std::vector<std::vector<std::vector<double>>> weights;
  //methods
  std::vector<double> calcOutputs(std::vector<double> inVec);
  void calcErrors(std::vector<int> targetVec);
  void updateNetwork(double learn_rate);
  // << operator overload
  friend std::ostream &operator<<(std::ostream& out, const mnist_network& net);
 public:
  // network constructor
  mnist_network(std::vector<int> layers);
  // network save
  void save_to_file(std::string filename);
  // network train
  void train(mnist_set ms, double learn_rate);
  // network test
  void test(mnist_set ms);
};

namespace static_network_utils {

  static std::vector<double> create_rand_arr(int size, float min_val, float max_val) {
    std::vector<double> v(size);
    for (auto it = v.begin(); it != v.end(); ++it) {
      *it = min_val + static_cast<double>(rand()) / (static_cast<float>(RAND_MAX/(max_val-min_val)));
    }
    return v;
  }

  static std::vector<std::vector<double>> create_rand_2d_arr(int rows, int cols, float min_val, float max_val) {
    std::vector<std::vector<double>> v(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        v[i][j] = min_val + static_cast<double>(rand()) / (static_cast<float>(RAND_MAX/(max_val-min_val)));
      }
    }
    return v;
  }

  static std::vector<std::vector<std::vector<double>>> init_weights(std::vector<int> layers, float min_val, float max_val) {
    std::vector<std::vector<std::vector<double>>> weights(layers.size());
    int layers_index = 0;
    for (int weights_index = 0; weights_index < weights.size() - 1; ++weights_index) {
      weights[weights_index] = create_rand_2d_arr(layers[0 + layers_index], layers[1 + layers_index], -1, 1);
      ++layers_index;
    }
    return weights;
  }

  static int max_element(std::vector<double> v) {
    int j = 0;
    int k = v[0];
    for (int i = 1;i<v.size();i++) {
      if (v[i] > k) {k = v[i];j = i;}
    }
    return j;
  }

  static double sigmoid(double x) {
    return 1 / (1 + pow(exp(1), -x));
  }

}

#endif //MNISTNETWORK__MNIST_NETWORK_H_
