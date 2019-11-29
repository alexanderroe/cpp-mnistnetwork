#include <iostream>
#include "src/mnist_network.h"
#include "src/mnist_set.h"

int main() {
  std::vector<int> v{784,350,100,10};
  mnist_network net(v);
  mnist_set trainset(0,1000,"/Users/alexander/CLionProjects/mnistnetwork/resources/train-images.idx3-ubyte",
                     "/Users/alexander/CLionProjects/mnistnetwork/resources/train-labels.idx1-ubyte");
  mnist_set testset(0,100,"/Users/alexander/CLionProjects/mnistnetwork/resources/t10k-images.idx3-ubyte",
                    "/Users/alexander/CLionProjects/mnistnetwork/resources/t10k-labels.idx1-ubyte");
  net.train(trainset, 0.1);
  net.test(testset);
}