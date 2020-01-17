#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>

using std::vector;

struct Network {

  // class variables
  int numLayers;
  vector<int> layers;
  vector<vector<double>> biases;
  vector<vector<double>> errors;
  vector<vector<double>> outputs;
  vector<vector<vector<double>>> weights;

  // init methods
  void initBiases();
  void initErrors();
  void initOutputs();
  void initWeights();

  // static methods
  static void genRand(vector<double>& v);
  static double sigmoid(double x);

 public:

  // constructors
  Network() = delete;
  explicit Network(const vector<int>& l);
//  explicit Network(const std::string& fileName);

  // feedforward
  void forward(vector<double>& input);
  void calcOutputs();

  // backprop and update
  void backward(const vector<int>& target, double learningRate);
  void calcErrors(const vector<int>& target);
  void updateNetwork(double learningRate);

  void train(vector<double>& input, const vector<int>& target, double learningRate);

  void save(const std::string& fileName);

  friend std::ostream& operator<<(std::ostream& out, const Network& net);

};