#include "Dataset.h"

Dataset::Dataset(const std::string &imagePath, const std::string &labelPath, int length) {

  if (length < 1 || length > kTrainSetLength) {
    throw std::invalid_argument("Bad index for Dataset constructor! Try 1 <= length <= 60000.");
  }

  std::ifstream imageStream(imagePath);
  std::ifstream labelStream(labelPath);

  if (!imageStream || ! labelStream) {
    throw std::runtime_error("Ifstream in Dataset constructor not properly initialized.");
  }

  imageStream.seekg(kImagesOffset + pow(kImageSize, 2));
  labelStream.seekg(kLabelsOffset);

  for (int i = 0; i < length; ++i) {

    std::vector<double> image(pow(kImageSize, 2));
    for (double& pixelValue : image)
      pixelValue = static_cast<double>(imageStream.get()) / kPixelValBlack;

    std::vector<int> label(kNumClasses, 0);
    label[labelStream.get()] = 1;

    Sample s(image, label);
    samples.push_back(s);
  }
}