import numpy as np
import skimage as sk
import torch

def single_pic_to_ndarray (imgPath, elType = np.float32):

    img = sk.io.imread(imgPath)
    img = img.astype(elType)
    img = np.moveaxis(img, 2, 0)
    img = img[np.newaxis, :]
    return img

def single_pic_to_4x_ndarray (imgPath, elType = np.float32):

    img = sk.io.imread(imgPath)
    img = img.astype(elType)
    img = sk.transform.rescale(img, 4., multichannel = True, anti_aliasing = True, mode = "constant")
    img = np.moveaxis(img, 2, 0)
    img = img[np.newaxis, :]
    img = img.astype(elType)
    return img
    
def save_ndarray_to_single_pic (array, imgPath):

    array = array[0, :]
    array = array.clip(0, 255)
    array = np.moveaxis(array, 0, 2)
    array = array.astype(np.uint8)
    sk.io.imsave(imgPath, array, quality = 95)

def twice_rescale_single_pic_to_ndarray (imgPath, elType = np.float32):

    #function that shrinks and restretches the picture with linear interpolation

    img = sk.io.imread(imgPath)
    img = img.astype(elType)

    #rounding picture size to multiple of four:
    modY, modX = img.shape[0:2]
    modY %= 4
    modX %= 4
    img = img[(0+int(modY/2)):(img.shape[0]-modY%2), (0+int(modX/2)):(img.shape[1]-modX%2)]

    #scaling down
    img = sk.transform.rescale(img, 0.25, multichannel = True, anti_aliasing = True, mode = "constant")

    #scaling back up
    img = sk.transform.rescale(img, 4, multichannel = True, anti_aliasing = True, mode = "constant")

    #final changes
    img = np.moveaxis(img, 2, 0)
    img = img[np.newaxis, :]
    img = img.astype(elType)

    return img

def crop_and_denoise_pic_to_ndarray (imgPath, kernel = 25, elType = np.float32, sigma = 0.8):

    #function that prepares the picture as a learning target

    img = sk.io.imread(imgPath)
    img = img.astype(elType)

    #rounding picture size to multiple of four:
    modY, modX = img.shape[0:2]
    modY %= 4
    modX %= 4

    #cropping away the pixels lost during the net execution:
    modY += kernel - 1
    modX += kernel - 1
    img = img[(0+int(modY/2)):(img.shape[0]-modY+int(modY/2)), (0+int(modX/2)):(img.shape[1]-modX+int(modX/2))]

    #denoising using Gaussian blur
    #img = sk.filters.gaussian(img, sigma = sigma, multichannel = True)

    #final changes
    img = np.moveaxis(img, 2, 0)
    img = img[np.newaxis, :]
    img = img.astype(elType)

    return img

def save_twice_rescaled_pic_to_ndarray (imgPath, arrPath, elType = np.float32):

    #function that loads a picture, converts it into a twice rescaled version, converts it
    #into a ndarray and saves it to disk (meant for parallelized conversion)

    img = twice_rescale_single_pic_to_ndarray(imgPath, elType = np.float32)
    np.save(arrPath, img)

def save_cropped_and_denoised_pic_to_ndarray (imgPath, arrPath, elType = np.float32):

    #function that loads a picture, denoises and crops it, converts it
    #into a ndarray and saves it to disk (meant for parallelized conversion)

    img = crop_and_denoise_pic_to_ndarray(imgPath, elType = np.float32)
    np.save(arrPath, img)

def single_pic_to_both_versions (imgPath, kernel = 25, elType = np.float32, sigma = 0.8):

    #function that takes a picture and returns a tuple of scaled and cropped versions

    img = sk.io.imread(imgPath)
    img = img.astype(elType)

    startShape = img.shape #debug

    #rounding picture size to multiple of four:
    modY, modX = img.shape[0:2]
    modY %= 4
    modX %= 4
    img = img[(0+int(modY/2)):(img.shape[0]-modY+int(modY/2)), (0+int(modX/2)):(img.shape[1]-modX+int(modX/2))]

    roundedShape = img.shape #debug

    #creating a scaled version
    #scaling down
    imgScaled = sk.transform.rescale(img, 0.25, multichannel = True, anti_aliasing = True, mode = "constant")

    #scaling back up
    imgScaled = sk.transform.rescale(imgScaled, 4, multichannel = True, anti_aliasing = True, mode = "constant")

    #final changes
    imgScaled = np.moveaxis(imgScaled, 2, 0)
    imgScaled = imgScaled[np.newaxis, :]
    imgScaled = imgScaled.astype(elType)

    #creating a cropped version
    #cropping away the pixels lost during the net execution:
    modY = kernel - 1
    modX = kernel - 1
    imgCropped = img[(0+int(modY/2)):(img.shape[0]-modY+int(modY/2)), (0+int(modX/2)):(img.shape[1]-modX+int(modX/2))]

    #denoising using Gaussian blur
    imgCropped = sk.filters.gaussian(imgCropped, sigma = sigma, multichannel = True)

    #final changes
    imgCropped = np.moveaxis(imgCropped, 2, 0)
    imgCropped = imgCropped[np.newaxis, :]
    imgCropped = imgCropped.astype(elType)

    s1 = imgScaled.shape
    s2 = imgCropped.shape

    if (s1[2] != s2[2] + kernel - 1 or s1[3] != s2[3] + kernel - 1):
        print(imgPath, "causes errors!")
        print(startShape)
        print(roundedShape)
        print(s1)
        print(s2)
    
    del img
    return (torch.as_tensor(imgScaled), torch.as_tensor(imgCropped))
