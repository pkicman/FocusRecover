#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <opencv2/highgui.hpp>

#include "InputData.h"
#include "myMat.h"
#include "utilities.h"
#include "FocusReconstructor.h"



int main(int argc, char** argv)
{
	if (argc != 3)
	{
		std::cerr << "Please define the input folder with using -f option" << std::endl;
		return 1;
	}
	std::string folder_pattern;
	if (std::string(argv[1]) == "-f")
		folder_pattern = argv[2];
	else
	{
		std::cerr << "Unknown parameter" << argv[1] << std::endl;
		return 1;
	}

	// algorithm parameters
	unsigned int blur_kernel_size = 3;
	unsigned int focus_kernel_size = 3;
	float ml_threshold = 8.0f;
	float focus_threshold = 32.0f;

	// Create object for handling input data
	InputData ID = InputData(folder_pattern, true);
	
	// Create object that contains the implemented algorithm
	FocusReconstructor FR = FocusReconstructor(blur_kernel_size, focus_kernel_size,
		ml_threshold, focus_threshold, ID.getTotalFramesCount());

	std::cout << "Loading " << ID.getTotalFramesCount() << " images" << std::endl;
	cv::Mat frame;
	
	while (ID.loadNextFrame())
	{
		// load the image and save for later processing		
		frame = ID.getNextFrame();
		FR.addImage(frame);
	}

	cv::Mat finalImg, depthMap;
	myMat<uchar> myfinalImg;
	myMat<float> mydepthMap;
	FR.processInputs();
	FR.reconstructFine(myfinalImg, mydepthMap);

	if (!myfinalImg.empty())
	{
		myfinalImg.toOpenCVMat8UC3(finalImg);
		mydepthMap.toOpenCVMat32FC1(depthMap);
		std::cout << "Processing completed" << std::endl;
		cv::imwrite("reconstructed_img.png", finalImg);
		cv::imwrite("depth_map.png", 255 * depthMap);

		cv::imshow("Final Image", finalImg);
		cv::imshow("Final depth map", depthMap);

		cv::waitKey(0);
	}
	else
	{
		std::cerr << "Processing unsuccessful" << std::endl;
	}
}