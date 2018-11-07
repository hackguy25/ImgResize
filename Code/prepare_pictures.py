#A script that prepares the pictures for training in parallel

import numpy as np
import multiprocessing as mp
import utility_functions as uf
import os

def prepare_scaled_pics (sourceFolder, destinationFolder):

    #reading the list of pictures
    picList = [i.replace(".jpg", "") for i in os.listdir(sourceFolder) if i.find(".jpg") > -1];

    #start of the multiprocess block
    with mp.Pool(processes = 12) as pool:

        #mapping the conversion function to the list of pictures
        argList = [(sourceFolder + i + ".jpg", destinationFolder + i + ".npy") for i in picList]
        pool.starmap(uf.save_twice_rescaled_pic_to_ndarray, argList)
        
        pool.wait()

def prepare_cropped_pics (sourceFolder, destinationFolder):

    #reading the list of pictures
    picList = [i.replace(".jpg", "") for i in os.listdir(sourceFolder) if i.find(".jpg") > -1];

    #start of the multiprocess block
    with mp.Pool(processes = 12) as pool:

        #mapping the conversion function to the list of pictures
        argList = [(sourceFolder + i + ".jpg", destinationFolder + i + ".npy") for i in picList]
        pool.starmap(uf.save_cropped_and_denoised_pic_to_ndarray, argList)
        
        pool.wait()
