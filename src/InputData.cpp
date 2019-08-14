#include "InputData.h"

InputData::InputData(int cam_id) {
	// Constructor for working directly with camera
	_source = 0;

}

InputData::InputData(std::string foldername, bool isVerbose) {
	// Constructor for loading images from a folder
	_source				= 1;
	_frame_available	= false;
	frame_count			= 0;
	folder_name			= foldername;
	_flagVerbose		= isVerbose;
	img_pattern			= foldername;
	// Automatically load all available files from the folder
	glob(img_pattern, _img_filenames);	
}

InputData::InputData() {
	this->_source = 2;
}


bool InputData::loadNextFrame() {

	switch (_source) {
	case 0:
		std::cout << "Not yet implemented" << std::endl;
		return false;
	case 1:
		if (loadImageFromFile())
		{
			_frame_available = true;
			return true;
		}
		break;
	case 2:
		std::cout << "Not yet implemented" << std::endl;
		return false;
	default:
		std::cout << "Unknown source" << std::endl;
		return false;
	}
	return false;
}

bool InputData::loadImageFromFile() {
	
	if (frame_count >= getTotalFramesCount())
		return false;

	if (_flagVerbose)
		std::cout << _img_filenames[frame_count] << std::endl;
		
	// check if the file exists
	std::ifstream myImage(_img_filenames[frame_count]);

	if (myImage.good())
	{		
		// load the image
		current_frame = imread(_img_filenames[frame_count], 1);
		frame_count++;
		return true;
	}
	else
	{		
		return false;
	}
}

cv::Mat InputData::getNextFrame() {
	if (_frame_available){
		_frame_available = false;
		return current_frame;
	}
	else {
		return cv::Mat();
	}
}
