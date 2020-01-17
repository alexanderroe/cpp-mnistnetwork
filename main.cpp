#include <sstream>
#include "src/Network.h"
#include "src/Dataset.cpp"

static void train(Network& n, const Dataset& t, double learningRate) {
  for (Dataset::Sample s : t.samples) {
    n.train(s.image, s.label, learningRate);
  }
}

static void test(Network& n, const Dataset& t) {
  int correct = 0;
  for (Dataset::Sample s : t.samples) {
    n.forward(s.image);
    int correctLabel = std::max_element(s.label.begin(), s.label.end()) - s.label.begin();
    int prediction = std::max_element(n.outputs[n.numLayers-1].begin(), n.outputs[n.numLayers-1].end()) - n.outputs[n.numLayers-1].begin();
    if (correctLabel == prediction) {
      ++correct;
    }
  }
  std::cout<< correct <<std::endl;
  std::cout<<t.samples.size()<<std::endl;
  std::cout << static_cast<double>(correct) / t.samples.size();
}

int main() {

  // define layer structure
  std::vector<int> layers({784,350,100,10});

  // construct network
  Network net(layers);

  // training data file paths
  std::string imagepath = "/Users/alexander/CLionProjects/mnistnetwork/resources/train-images.idx3-ubyte";
  std::string labelpath = "/Users/alexander/CLionProjects/mnistnetwork/resources/train-labels.idx1-ubyte";

  // construct trainset
  Dataset trainset(imagepath, labelpath, 6);

  // train network with trainset and learning rate
  train(net, trainset, 0.5);

  // save network
  net.save("test.txt");

//  // testing data file paths
//  std::string imagetest = "/Users/alexander/CLionProjects/mnistnetwork/resources/t10k-images.idx3-ubyte";
//  std::string labeltest = "/Users/alexander/CLionProjects/mnistnetwork/resources/t10k-labels.idx1-ubyte";
//
//  // construct testset
//  Dataset testset(imagetest, labeltest, 0, 5000);
//
//  //test network with testset
//  test(net, testset);
}

//.5, 60000, 5000 = .9136
//hyperparam.: lr, trainset length, testset length, (data source), layers