#include <iostream>
#include <utility>
#include <vector>
#include <cmath>
#include <fstream>

#define kImageSize 28
#define kNumClasses 10
#define kImagesOffset 16
#define kLabelsOffset 8
#define kTrainSetLength 60000
#define kPixelValBlack 255

struct Dataset {

  // inner data sample struct
  struct Sample {
    // struct fields
    std::vector<double> image;
    std::vector<int> label;
    // constructor
    Sample(std::vector<double>  image, std::vector<int>  label) : image(std::move(image)), label(std::move(label)) {}
  };

  // class field
  std::vector<Sample> samples;

  // constructor
  Dataset(const std::string& imagePath, const std::string& labelPath, int length);
};

