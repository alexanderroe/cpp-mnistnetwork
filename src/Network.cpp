#include "Network.h"

// init methods

void Network::initBiases() {
  for (int layerSize : layers) {
    vector<double> v(layerSize);
    genRand(v);
    biases.push_back(v);
  }
}
void Network::initErrors() {
  for (int layerSize : layers) {
    vector<double> v(layerSize, 0);
    errors.push_back(v);
  }
}
void Network::initOutputs() {
  for (int layerSize : layers) {
    vector<double> v(layerSize, 0);
    outputs.push_back(v);
  }
}
void Network::initWeights() {
  for (int layer = 0; layer < numLayers-1; ++layer) {
    vector<vector<double>> v(layers[layer], vector<double>(layers[layer+1]));
    for (vector<double>& vec : v) {
      genRand(vec);
    }
    weights.push_back(v);
  }
}

// constructors

Network::Network(const vector<int> &l) {
  numLayers = l.size();
  layers = l;
  initBiases();
  initErrors();
  initOutputs();
  initWeights();
}
//Network::Network(const std::string &fileName) {
////  initErrors();
////  initOutputs();
////  std::ifstream file;
////  file.open(fileName);
//
//}

// feedforward

void Network::forward(vector<double> &input) {
  outputs[0] = input;
  calcOutputs();
}
void Network::calcOutputs() {
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

// backprop

void Network::backward(const vector<int> &target, double learningRate) {
  calcErrors(target);
  updateNetwork(learningRate);
}
void Network::calcErrors(const vector<int> &target) {
  for (int neuron = 0; neuron < layers[numLayers-1]; ++neuron) {
    errors[numLayers-1][neuron] = (outputs[numLayers-1][neuron] - target[neuron]) * outputs[numLayers-1][neuron] * (1 - outputs[numLayers-1][neuron]);
  }

  for (int layer = numLayers-2; layer > 0; --layer) {
    for (int neuron = 0; neuron < layers[layer]; ++neuron) {
      double sum = 0;
      for (int nextNeuron = 0; nextNeuron < layers[layer+1]; ++nextNeuron) {
        sum += weights[layer][neuron][nextNeuron] * errors[layer+1][nextNeuron];
      }
      errors[layer][neuron] = sum * outputs[layer][neuron] * (1 - outputs[layer][neuron]);
    }
  }
}
void Network::updateNetwork(double learningRate) {
  for (int layer = 1; layer < numLayers; ++layer) {
    for (int neuron = 0; neuron < layers[layer]; ++neuron) {
      biases[layer][neuron] -= learningRate * errors[layer][neuron];
      for (int prevNeuron = 0; prevNeuron < layers[layer - 1]; ++prevNeuron) {
        weights[layer - 1][prevNeuron][neuron] -= learningRate * errors[layer][neuron] * outputs[layer - 1][prevNeuron];
      }
    }
  }
}

// train & save

void Network::train(vector<double> &input, const vector<int> &target, double learningRate) {
  forward(input);
  backward(target, learningRate);
}
void Network::save(const std::string &fileName) {
  std::ofstream file;
  file.open(fileName);
  file << "n" << numLayers << std::endl;
  file << "l";
  for (int i : layers) {
    file << i << " ";
  }
  file << "\nb";
  for (vector<double>& vec : biases) {
    for (double& d : vec) {
      file << d << " ";
    }
    file << "\n";
  }
  file << "\nw";
  for (vector<vector<double>>& mat : weights) {
    for (vector<double>& vec : mat) {
      for (double& d : vec) {
        file << d << " ";
      }
      file << "\n";
    }
    file << "\n";
  }
  file.close();
}

// static helpers

void Network::genRand(vector<double> &v) {
  std::random_device rd{};
  std::mt19937 engine {rd()};
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  generate(v.begin(), v.end(), [&dist, &engine](){
    return dist(engine);
  });
}
double Network::sigmoid(double x) {
  return 1 / (1 + exp(-x));
}

// << overload

std::ostream &operator<<(std::ostream &out, const Network &net) {
  out << "Number of layers: " << net.numLayers << std::endl;
  int layerIndex = 1;
  for (int layer : net.layers) {
    std::cout << "Layer " << layerIndex << ": " << layer << std::endl;
    ++layerIndex;
  }
  return out;
}