#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>

struct Network {

  // class variables
  int numLayers;
  std::vector<int> layers;
  std::vector<std::vector<double>> biases;
  std::vector<std::vector<double>> errors;
  std::vector<std::vector<double>> outputs;
  std::vector<std::vector<std::vector<double>>> weights;  // cleanup with "using std::vector" ?

  // default constructor
  Network() = delete; // is this a good idea? suppress this form of instantiation

  // init methods
  void initBiases() {
    for (int layerSize : layers) {
      std::vector<double> v(layerSize);
      genRand(v);
      biases.push_back(v);
    }
  }
  void initErrors() {
    for (int layerSize : layers) {
      std::vector<double> v(layerSize, 0);
      errors.push_back(v);
    }
  }
  void initOutputs() {
    for (int layerSize : layers) {
      std::vector<double> v(layerSize, 0);
      outputs.push_back(v);
    }
  }
  void initWeights() {
    for (int layer = 0; layer < numLayers-1; ++layer) {
      std::vector<std::vector<double>> v(layers[layer], std::vector<double>(layers[layer+1]));
      for (std::vector<double>& vec : v) {
        genRand(vec);
      }
      weights.push_back(v);
    }
  }

  // static rng method
  static void genRand(std::vector<double>& v) {
    std::random_device rd{};
    std::mt19937 engine {rd()};
    std::uniform_real_distribution<double> dist{-1.0, 1.0};
    generate(v.begin(), v.end(), [&dist, &engine](){
      return dist(engine);
    });
  }

  // static activation function
  static double sigmoid(double x) {
    return 1 / (1 + exp(-x));
  }

 public:

  explicit Network(const std::vector<int>& l) {
    numLayers = l.size();
    layers = l;
    initBiases();
    initErrors();
    initOutputs();
    initWeights();
  }

  Network(const std::string& fileName) {

  }

  void forward(const std::vector<double>& input) {
    outputs[0] = input;
    calcOutputs();
  }

  void calcOutputs() {
    for (int layer = 1; layer < numLayers; ++layer) {
      for (int neuron = 0; neuron < layers[layer]; ++neuron) {
        double weightedSum = biases[layer][neuron];
        for (int prevNeuron = 0; prevNeuron < layers[layer-1]; ++prevNeuron) {
          weightedSum += outputs[layer-1][prevNeuron] * weights[layer-1][prevNeuron][neuron];
        }
        outputs[layer][neuron] = sigmoid(weightedSum);
      }
    }
  }

  void backward(const std::vector<int>& target, double learningRate) {
    calcErrors(target);
    std::cout<<"";
    updateNetwork(learningRate);
    std::cout<<"";
  }

  void calcErrors(const std::vector<int>& target) {
    for (int neuron = 0; neuron < layers[numLayers-1]; ++neuron) {
      errors[numLayers-1][neuron] = (outputs[numLayers-1][neuron] - target[neuron]) * outputs[numLayers-1][neuron] * (1 - outputs[numLayers-1][neuron]);
    }
    for (int layer = numLayers-2; layer > 0; --layer) {
      for (int neuron = 0; neuron < layers[layer]; ++neuron) {
        double sum = 0;
        for (int nextNeuron = 0; nextNeuron < layers[layer+1]; ++nextNeuron) {
          sum += weights[layer][neuron][nextNeuron] * errors[layer+1][nextNeuron];
        }
        errors[layer][neuron] = sum * outputs[numLayers-1][neuron] * (1 - outputs[numLayers-1][neuron]);
      }
    }
  }

  void updateNetwork(double learningRate) {
    for (int layer = 1; layer < numLayers; ++layer) {
      for (int neuron = 0; neuron < layers[layer]; ++neuron) {
        biases[layer][neuron] -= learningRate * errors[layer][neuron];
        for (int prevNeuron = 0; prevNeuron < layers[layer - 1]; ++prevNeuron) {
          weights[layer - 1][prevNeuron][neuron] -= learningRate * errors[layer][neuron] * outputs[layer - 1][prevNeuron];
        }
      }
    }
  }

  void train(const std::vector<double>& input, const std::vector<int>& target, double learningRate) {
    forward(input);
    std::cout<<"";
    backward(target, learningRate);
    std::cout<<"";
  }

  void save(const std::string& fileName) {
    std::ofstream file;
    file.open(fileName);
    file << "b";
    for (std::vector<double>& vec : biases) {
      for (double& d : vec) {
        file << d << " ";
      }
      file << "\n";
    }
    file << "\nw";
    for (std::vector<std::vector<double>>& mat : weights) {
      for (std::vector<double>& vec : mat) {
        for (double& d : vec) {
          file << d << " ";
        }
        file << "\n";
      }
      file << "\n";
    }
    file.close();
  }

  friend std::ostream& operator<<(std::ostream& out, const Network& net) {
    out << "Number of layers: " << net.numLayers << std::endl;
    int layerIndex = 1;
    for (auto it = net.layers.begin(); it != net.layers.end(); ++it) {
      std::cout << "Layer " << layerIndex << ": " << *it << std::endl;
      ++layerIndex;
    }
    return out;
  }

};