#include "./src/Network.cpp"
#include "./src/Trainset.cpp"

static void train(Network& n, const Trainset& t) {
  for (Trainset::Sample s : t.samples) {
    n.train(s.image, s.label, 0.01);
  }
}

static void test(Network& n, const Trainset& t) {
  int correct = 0;
  for (Trainset::Sample s : t.samples) {
    n.forward(s.image);
    int correctLabel = std::max_element(s.label.begin(), s.label.end()) - s.label.begin();
    int prediction = std::max_element(n.outputs[n.numLayers-1].begin(), n.outputs[n.numLayers-1].end()) - n.outputs[n.numLayers-1].begin();
    if (correctLabel == prediction) {
      ++correct;
    }
  }
  std::cout << static_cast<double>(correct) / t.samples.size();
}


int main() {
  std::vector<int> l({784,350,100,10});
  Network n(l);
  std::string imagepath = "/Users/alexander/CLionProjects/mnistnetwork/resources/train-images.idx3-ubyte";
  std::string labelpath = "/Users/alexander/CLionProjects/mnistnetwork/resources/train-labels.idx1-ubyte";
  Trainset t(imagepath, labelpath,0, 1000);
  train(n, t);
  test(n, t);
}