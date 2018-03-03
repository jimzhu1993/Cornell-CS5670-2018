import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')

import cv2
import numpy as np
import math

def apply_kernel(img, kernel, is_convolution):
    img_row, img_col = img.shape
    kernel_row, kernel_col = kernel.shape
    # given kernel dimension is odd number
    k_row_center = int(kernel_row / 2)
    k_col_center = int(kernel_col / 2)
    # augmenting the img to make computation easier
    img_aug = np.pad(img, ((k_row_center, k_row_center), (k_col_center, k_col_center)), 'constant')
    if not is_convolution:
        kernel_filter = kernel.reshape((1, -1))[0]
    else:
        kernel_filter = kernel.reshape((1, -1))[0][::-1]

    G = np.zeros((img_row, img_col), dtype=img.dtype)
    for i in range(img_row):
        for j in range(img_col):
            feature_matrix = img_aug[i: i + kernel_row, j: j + kernel_col].reshape((1, -1))[0]
            G[i, j] = feature_matrix.dot(kernel_filter)
    return G

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN

    if kernel.shape[0] * kernel.shape[1] & 1 != 1:
        raise Exception("kernel dimension is not odd")
    else:
        if len(img.shape) == 2:
            return apply_kernel(img, kernel, False)
        else:
            G = np.zeros(img.shape, dtype = img.dtype)
            for i in range(img.shape[2]):
                G[:, :, i] = apply_kernel(img[:, :, i], kernel, False)
    return G
    # TODO-BLOCK-END

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN

    if kernel.shape[0] * kernel.shape[1] & 1 != 1:
        raise Exception("kernel dimension is not odd")
    else:
        if len(img.shape) == 2:
            return apply_kernel(img, kernel, True)
        else:
            G = np.zeros(img.shape, dtype= img.dtype)
            for i in range(img.shape[2]):
                G[:, :, i] = apply_kernel(img[:, :, i], kernel, True)
    return G
    # TODO-BLOCK-END

def gaussian_blur_kernel_2d(sigma, width, height):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions width x height such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # TODO-BLOCK-BEGIN

    # rounds to the next odd integer
    if width & 1 != 1:
        width = width + 1
    if height & 1 != 1:
        height = height + 1

    H = np.zeros((width, height))
    k_row_center = int(width / 2)
    k_col_center = int(height / 2)

    for i in range(width):
        for j in range(height):
            x = (i - k_row_center)
            y = (j - k_col_center)
            H[i, j] = (1.0 / (2.0 * math.pi * sigma ** 2)) * (math.e ** (-(x ** 2 + y ** 2) / (2.0 * sigma ** 2)))

    return H / H.sum()

    # TODO-BLOCK-END

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN

    gaussian_kernel = gaussian_blur_kernel_2d(sigma, size, size)
    return convolve_2d(img, gaussian_kernel)

    # TODO-BLOCK-END

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN

    return img - low_pass(img, sigma, size)

    # TODO-BLOCK-END

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)