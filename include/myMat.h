#pragma once
#include <iostream>
#include <cstring>
#include <opencv2/core.hpp>

typedef unsigned char uchar;
typedef unsigned int uint;

template<typename T> class myMat;

template <typename S>
bool operator==(const myMat<S> &lhs, const myMat<S> &rhs);


template<typename T>
class myMat {
private:
	uint m_nrows, m_ncols, m_nchannels;
	T *m_data;

	void init(uint r, uint c, uint ch);

public:
	myMat() : m_nrows(0), m_ncols(0), m_nchannels(0), m_data(nullptr) {}

	myMat(uint in_nrows, uint in_ncols, uint in_nchannels) :
		m_nrows(in_nrows), m_ncols(in_ncols), m_nchannels(in_nchannels) {
		if (m_nrows != 0 && m_ncols != 0 && m_nchannels != 0)
			m_data = new T[m_nrows*m_ncols*m_nchannels];
		else
			m_data = nullptr;
	}

	// copy constructor
	myMat(const myMat<T> &rhs) : m_nrows(rhs.m_nrows), m_ncols(rhs.m_ncols), m_nchannels(rhs.m_nchannels) {
		if (m_nrows != 0 && m_ncols != 0 && m_nchannels != 0)
		{
			m_data = new T[m_nrows*m_ncols*m_nchannels];
			memcpy(m_data, rhs.m_data, sizeof(T)*m_nrows*m_ncols*m_nchannels);
		}
		else
			m_data = nullptr;

	}

	// Construct myMat directly from an OpenCV Mat
	myMat(const cv::Mat &inMat) { fromOpenCVMat(inMat); }

	// destructor
	~myMat() {	delete[] m_data; }

	// check if matrix is empty
	bool empty() { return m_nrows == 0 || m_ncols == 0 || m_nchannels == 0; }

	// Performs a deep copy of OpenCV Mat class
	void fromOpenCVMat(const cv::Mat &inMat);

	// Exports matrix to OpenCV Mat class
	void toOpenCVMat8UC3(cv::Mat &outMat) const;
	void toOpenCVMat8UC1(cv::Mat &outMat) const;
	void toOpenCVMat32FC1(cv::Mat &outMat) const;

	// access matrix number of rows
	uint nrows() const { return m_nrows; }
	// access matrix number of columns
	uint ncols() const { return m_ncols; }
	// access matrix number of channels
	uint nchannels() const { return m_nchannels; }

	// returns pointer to the first channel element at position (i, j)
	T* at(uint i, uint j) const { return m_data + i*m_ncols*m_nchannels + j; }

	// returns pointer to the first element of the ith row
	T* rowPtr(uint i) const { return m_data + i*m_ncols*m_nchannels;	}

	// Fill all elements with a scalar value
	void fill(T scalar);

	// return sum of all elements in the matrix
	float sumAll() const;

	// print all the elements of the matrix to console
	void debugPrint() const;

	myMat<T>& operator=(const myMat<T> &rhs);

	friend bool operator==<T>(const myMat<T> &lhs, const myMat<T> &rhs);

};

template <typename T>
void myMat<T>::init(uint r, uint c, uint ch) {
	m_nrows = r;
	m_ncols = c;
	m_nchannels = ch;
	m_data = new T[m_nrows*m_ncols*m_nchannels];
}

template <typename T>
void myMat<T>::fromOpenCVMat(const cv::Mat &inMat) {

	// if allocated and same size - do not allocate
	// if not allocated - allocate
	// if allocated but incompatible size - free the memory and reallocate
	//if (m_data != nullptr)
	//{
	//	if (m_nrows != inMat.rows || m_ncols != inMat.cols || m_nchannels != inMat.channels())
	//	{
	//		delete[] m_data;
	//		init(inMat.rows, inMat.cols, inMat.channels());
	//	}
	//}
	//else
	//{
		init(inMat.rows, inMat.cols, inMat.channels());
	//}

	// copy all matrix elements
	if (m_nchannels == 1)
	{
		const T *r;
		for (uint i = 0; i < m_nrows; ++i)
		{
			r = inMat.ptr<T>(i);
			for (uint j = 0; j < m_ncols; ++j)
					at(i, j)[0] = r[j];
		}
	}
	if (m_nchannels == 3)
	{
		const cv::Vec<T, 3> *r;
		for (uint i = 0; i < m_nrows; ++i)
		{
			r = inMat.ptr<cv::Vec<T, 3>>(i);
			for (uint j = 0, k = 0; j < m_ncols; ++j, k += m_nchannels)
			{
				at(i, k)[0] = r[j][0];
				at(i, k)[1] = r[j][1];
				at(i, k)[2] = r[j][2];
			}
		}
	}
}

template <typename T>
void myMat<T>::toOpenCVMat8UC3(cv::Mat &outMat) const {

	if (m_nchannels != 3) return;

	outMat = cv::Mat(m_nrows, m_ncols, CV_8UC3);
	cv::Vec<T, 3> *r;
	for (uint i = 0; i < m_nrows; ++i)
	{
		r = outMat.ptr<cv::Vec<T, 3>>(i);
		for (uint j = 0, k = 0; j < m_ncols; ++j, k += m_nchannels)
		{
			r[j][0] = at(i, k)[0];
			r[j][1] = at(i, k)[1];
			r[j][2] = at(i, k)[2];
		}
	}
}

template <typename T>
void myMat<T>::toOpenCVMat8UC1(cv::Mat &outMat) const {

	if (m_nchannels != 1) return;

	outMat = cv::Mat(m_nrows, m_ncols, CV_8UC1);
	T *r;
	for (uint i = 0; i < m_nrows; ++i)
	{
		r = outMat.ptr<T>(i);
		for (uint j = 0; j < m_ncols; ++j)
		{
			r[j] = at(i, j)[0];
		}
	}
}

template <typename T>
void myMat<T>::toOpenCVMat32FC1(cv::Mat &outMat) const {

	if (m_nchannels != 1) return;

	outMat = cv::Mat(m_nrows, m_ncols, CV_32FC1);
	T *r;
	for (uint i = 0; i < m_nrows; ++i)
	{
		r = outMat.ptr<T>(i);
		for (uint j = 0; j < m_ncols; ++j)
		{
			r[j] = at(i, j)[0];
		}
	}
}

template <typename T>
void myMat<T>::fill(T scalar) {
	for (uint i = 0; i < m_nrows*m_ncols*m_nchannels; ++i)
		m_data[i] = scalar;
}

template <typename T>
float myMat<T>::sumAll() const {
	float sum = 0.0f;
	for (uint i = 0; i < m_nrows*m_ncols*m_nchannels; ++i)
		sum += m_data[i];
	return sum;
}

template <typename T>
void myMat<T>::debugPrint() const {
	std::cout << "myMat: " << m_nrows << "x" << m_ncols << " (" << m_nchannels << " channels)" << std::endl;

	for (uint i = 0; i < m_nrows; ++i)
	{
		std::cout << "Row " << i << ": ";
		for (uint j = 0; j < m_ncols*m_nchannels; j += m_nchannels)
		{
			if (m_nchannels == 1)
				std::cout << (float)m_data[i*m_ncols + j] << " ";
			if (m_nchannels == 3)
			{
				std::cout << (float)m_data[i*m_ncols*m_nchannels + j] << " ";
				std::cout << (float)m_data[i*m_ncols*m_nchannels + j + 1] << " ";
				std::cout << (float)m_data[i*m_ncols*m_nchannels + j + 2] << " ";
				std::cout << " ";
			}
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

template <typename T>
myMat<T>& myMat<T>::operator=(const myMat<T> &rhs) {
	if (this == &rhs) return *this;

	// if any of the sizes is different - free the previously allocated memory
	if (m_data == nullptr || m_nrows != rhs.m_nrows || m_ncols != rhs.m_ncols || rhs.m_nchannels)
	{
		delete[] m_data;
		init(rhs.m_nrows, rhs.m_ncols, rhs.m_nchannels);
	}
	// if the previous matrix was of the same size - there will be no reallocation
	memcpy(m_data, rhs.m_data, sizeof(T)*m_nrows*m_ncols*m_nchannels);
	return *this;
}

template <typename S>
bool operator==(const myMat<S> &lhs, const myMat<S> &rhs) {
	if (lhs.m_nrows != rhs.m_nrows || lhs.m_ncols != rhs.m_ncols || lhs.m_nchannels != rhs.m_nchannels)
		return false;
	for (uint i = 0; i < lhs.m_nrows*lhs.m_ncols*lhs.m_nchannels; ++i)
		if (lhs.m_data[i] != rhs.m_data[i])
			return false;
	return true;
}