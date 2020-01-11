#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

class Trainset {
  //constants
  static constexpr int kImageSize = 28;
  static constexpr int kNumClasses = 10;
  static constexpr int kImagesMagicNum = 16;
  static constexpr int kLabelsMagicNum = 8;
  // class variables
  std::vector<std::vector<double>> images;
  std::vector<std::vector<int>> labels;

 public:
  Trainset(std::ifstream& imageStream, std::ifstream& labelStream, int startIndex, int length) : images(length, std::vector<double>(pow(kImageSize, 2))), labels(length) {

    if (startIndex < 0) {
      throw std::invalid_argument("Bad startIndex for Trainset constructor!");
    }
    if (length < 1 || length > 60000) {
      throw std::invalid_argument("Bad index for Trainset constructor! Try 1 <= length <= 60000.");
    }

    imageStream.seekg(kImagesMagicNum + (startIndex * pow(kImageSize, 2)));
    labelStream.seekg(kLabelsMagicNum + startIndex);

    for (int i = 0; i < length; ++i) {

      std::vector<double> v(pow(kImageSize, 2));
      for (double& d : v) {
        d = static_cast<double>(imageStream.get()) / 255;
      }

      std::vector<int> v2(kNumClasses, 0);
      v2[labelStream.get()] = 1;

      images.push_back(v);
      labels.push_back(v2);
    }
  }
};

int main() {
  std::ifstream imagedata("/Users/alexander/CLionProjects/mnistnetwork/resources/train-images.idx3-ubyte");
  std::ifstream labeldata("/Users/alexander/CLionProjects/mnistnetwork/resources/train-labels.idx1-ubyte");
  Trainset t(imagedata, labeldata, 0, 10);
  std::cin.get();
}


/*
 * #include "mnist_set.h"
#include <fstream>
#include <cmath>

// 0-indexed, first inclusive, last exclusive
mnist_set::mnist_set(int first_index, int last_index, const std::string& image_data_file_path, const std::string& label_data_file_path) {

  if (first_index < 0) throw std::invalid_argument("first index must be at least 0");
  if (last_index > TRAINING_SET_SIZE) throw std::invalid_argument("last index must be at most 59999");

  std::ifstream imagedata(image_data_file_path, std::ios::binary);
  std::ifstream labeldata(label_data_file_path, std::ios::binary);
  imagedata.seekg(IMAGESET_MAGIC_NUM);
  labeldata.seekg(LABELSET_MAGIC_NUM);

  std::vector<image> images(last_index - first_index);
  for (auto &image : images) {
    std::vector<double> v(pow(IMAGE_SIZE, 2));
    for (int i = 0; i < pow(IMAGE_SIZE, 2); ++i) {
      v[i] = static_cast<double>(imagedata.get()) / 255;
    }
    image.data = v;
    image.label = labeldata.get();
  }
  set = images;
}

void mnist_set::printImage(const std::string& imagefile) {
  std::ifstream imagedata(imagefile, std::ios::binary);
  imagedata.seekg(16);
  for (int i = 0; i < 7840; ++i) {
    if (i%28==0) {
      std::cout<<"\n";
    }
    if (imagedata.get() == 0)
      std::cout<< 0;
    else
      std::cout<<1;
  }

}
std::ostream &operator<<(std::ostream &out, const mnist_set &ms) {
  out<<ms.set[0].data.size()<<std::endl;
  for (image i : ms.set) {
    for (int j = 0;j<784;j++) {     //magic number from network header
      if (j%28==0) out<<"\n";       //magic number
      if (i.data[j] > 0) out<<1;
      else out<<0;
    }
    out<<"\n"<<i.label<<std::endl;
  }
  return out;
}

//4 32-bit integers: magic number, num samples, rows, cols
 */