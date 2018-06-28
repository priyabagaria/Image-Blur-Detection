# Image-Blur-Detection
Classification of Blurred and Non-Blurred Images  

**CERTH Image Blur Dataset**


> E. Mavridaki, V. Mezaris, "No-Reference blur assessment in natural images using Fourier transform and spatial pyramids", Proc. IEEE International Conference on Image Processing (ICIP 2014), Paris, France, October 2014.


The dataset consists of undistorted, naturally-blurred and artificially-blurred images for image quality
assessment purposes.
Download the dataset from here:
http://mklab.iti.gr/files/imageblur/CERTH_ImageBlurDataset.zip

Unzip and load the files into a directory named **CERTH_ImageBlurDataset**



---

## Variation of the Laplacian
**Using OpenCV2**

This method yielded an accuracy of **87.29%**

`cv2.Laplacian(img, cv2.CV_64F).var()`

The Laplacian Operator is applied on each of the images. 
The variation of the result is calculated.
If the variation is below the threshold, 400 in this case, the image is classified as blurry.
Otherwise, it is classified as non-blurry.


To run this model:

`python OpenCV_var.py`

