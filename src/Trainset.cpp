#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

struct Trainset { //class or struct?

  // inner struct
  struct Sample {  //should be inner or separate struct?
    // struct fields
    std::vector<double> image;
    std::vector<int> label;
    // constructor
    Sample(const std::vector<double>& image, const std::vector<int>& label) : image(image), label(label) {}
  };

  // constants
  static constexpr int kImageSize = 28;
  static constexpr int kNumClasses = 10;
  static constexpr int kImagesMagicNum = 16;
  static constexpr int kLabelsMagicNum = 8;
  // class variables
  std::vector<Sample> samples;

  // constructor
  Trainset(std::string imagePath, std::string labelPath, int startIndex, int length) {

    if (startIndex < 0) {
      throw std::invalid_argument("Bad startIndex for Trainset constructor!");
    }
    if (length < 1 || length > 60000) {
      throw std::invalid_argument("Bad index for Trainset constructor! Try 1 <= length <= 60000.");
    }

    std::ifstream imageStream(imagePath);
    std::ifstream labelStream(labelPath);

    imageStream.seekg(kImagesMagicNum + (startIndex * pow(kImageSize, 2)));
    labelStream.seekg(kLabelsMagicNum + startIndex);

    for (int i = 0; i < length; ++i) {

      std::vector<double> v(pow(kImageSize, 2));
      for (double& d : v)
        d = static_cast<double>(imageStream.get()) / 255;

      std::vector<int> v2(kNumClasses, 0);
      v2[labelStream.get()] = 1;

      Sample s(v, v2);    //better local var names
      samples.push_back(s);
    }
  }
};

