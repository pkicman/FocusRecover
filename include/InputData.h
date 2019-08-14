#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>


class InputData {

public:
	/// Use camera as a source
	InputData(int cam_id);

	/// Use folder with images as a source
	InputData(std::string foldername, bool isVerbose);

	/// Use video file as a source
	InputData();
	bool loadImageFromFile();
	/// Load next frame from the stream
	bool loadNextFrame();

	int getTotalFramesCount() const { return _img_filenames.size(); };

	bool isFrameAvailable() const { return _frame_available; };

	cv::Mat getNextFrame();

private:
	bool _flagVerbose;
	bool _frame_available;
	unsigned short int _source;
	unsigned int frame_count;
	std::vector<cv::String> _img_filenames;

	cv::Mat current_frame;

	std::string folder_name;
	cv::String img_pattern;

};