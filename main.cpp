#include "./src/Network.cpp"
#include "./src/Trainset.cpp"

static void train(Network& n, const Trainset& set) {
  for (Trainset::Sample s : set.samples) {
    n.train(s.image, s.label, 0.01);
  }
}

int main() {
  std::vector<int> l({784,350,100,10});
  Network n(l);
  Trainset t("/Users/alexander/CLionProjects/mnistnetwork/resources/train-images.idx3-ubyte",
           "/Users/alexander/CLionProjects/mnistnetwork/resources/train-labels.idx1-ubyte",
           0, 100);
  train(n, t);
  n.forward(t.samples[0].image);
  std::cin.get();
}