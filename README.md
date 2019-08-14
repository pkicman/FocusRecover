# Intro
The goal of this app is to reconstruct a sharp image of a bug from a set of narrowly focused images and in the same time to build a depth map. The input images are stored in the "bug" folder.

# Building the code
OpenCV is required to build and run the code. cmake should take care of finding the appropriate installation of OpenCV, but should you have any trouble, just modify this line in CMakeLists.txt:
```set( "OpenCV_DIR" "F:\\workspace\\libraries\\opencv_3_4\\opencv_mybuild\\install\\" )```
to point cmake directly to your OpenCV installation folder.

The app was tested on Windows 8.1 with Opencv 3.4 and Ubuntu 18.04 with OpenCV 3.2.

# Running the code
To run the code simply type in the console:
```./FocusRecover -f "path/to/images/b_bigbug*.png"```

# Short discussion
The executes in three major steps:
1. Loading all images
2. Computing the focus score for every pixel in every image
3. For every pixel find image in which the pixel has best focus, from that - recover the image values and build the depth map

There are multiple ways to compute a focus metric for an image. A great overview of the available approaches is provided in [1]. In the application I decided to use a Modified Laplacian metric coming from [2]. One modification wrt. the original method is use of using quadratic function (instead of fitting Gaussian function) to estimate the fine depth of pixel. This deviation was mostly due to reduce slightly a complexity of the task and to speed up the coding. According to the [1] the selected method is one of the most accurate ones, while at the same time fairly fast. The major drawback of the method is sensitivity to the noise in the image. The compensate for the sensitivity of the selected method to the noise the image is initially blurred with a small kernel. This obviously reduces the ability of reconstructing fine details. On the other hand my tests shown that when the images are not blurred the background is reconstructed with a very high level of noise.

Two reconstructions can be computed by a FocusReconstructor class:

- coarse - no fitting of the quadratic function. Used initially for testing and debugging, to use it one must recompile the code
- fine (default option) - the depth of the image is computed fitting a quadratic function into the three points surrounding the highest scoring index.




# Results

Finally the following result was obtained:

![](https://github.com/pkicman/FocusRecover/doc/reconstructed.png)



![](https://github.com/pkicman/FocusRecover/doc/depth_map.png)




# Useful resources:
1. Petruz, S, et.al., Analysis of focus measure operators for shape-from-focus, Pattern Recognition, 2013
2. Nayar, S.K., Shape from Focus, 1989
3. https://stackoverflow.com/questions/7765810/is-there-a-way-to-detect-if-an-image-is-blurry