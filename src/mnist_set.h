#ifndef MNISTNETWORK_SRC_MNIST_SET_H_
#define MNISTNETWORK_SRC_MNIST_SET_H_
#include <iostream>
#include <vector>

// image structure
struct image {
  std::vector<double> data;
  int label;
};

// class definition
struct mnist_set {
  //const
  static constexpr int IMAGE_SIZE = 28;
  static constexpr int TRAINING_SET_SIZE = 59999;
  static constexpr int IMAGESET_MAGIC_NUM = 16;
  static constexpr int LABELSET_MAGIC_NUM = 8;
  // set variable
  std::vector<image> set;
  // << operator overload
  friend std::ostream &operator<<(std::ostream &out, const mnist_set& ms);
  // constructor
  mnist_set(int first_index, int last_index, const std::string &image_data_file_path, const std::string &label_data_file_path);
  // utility method
  static void printImage(const std::string& imagefile);
};

#endif //MNISTNETWORK_SRC_MNIST_SET_H_
