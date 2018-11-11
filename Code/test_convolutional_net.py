#A simple script that executes the net on a folder of pictures.

import single_layer_convolutional_net as myNet
import utility_functions as uf
import torch
import sys
import os

def createDir (dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print ("Error creating directory " + dir + ".")
        sys.exit(-1)

def execute (sourceFolder, outFolder, state = "current", descriptive = True, kernel = 25):

    #creating an instance of the net, sent to the GPU
    if descriptive: print("Initializing the net.")

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net, opt, crit = myNet.loadNet("./States/" + state + "/", dev, kernel = kernel)

    if descriptive: print("Initialized.")

    #reading the list of pictures in the source folder
    picList = [i.replace(".jpg", "") for i in os.listdir(sourceFolder) if i.find(".jpg") > -1];

    if (len(picList) == 0):

        if descriptive: print("No pictures found.")
    else:

        if descriptive: print("Number of pictures: " + str(len(picList)) + ".")
        
        createDir(outFolder + state + "/")

        #converting pictures, one by one
        for n, i in enumerate(picList):

            #convert current picture to an upscaled ndarray
            imgInput = uf.single_pic_to_4x_ndarray(sourceFolder + i + ".jpg")
            #convert ndarray to a tensor
            imgInput = torch.as_tensor(imgInput, device=dev)
            #execute the net on the image tensor
            imgInput = net(imgInput)
            #convert the tensor back to a ndarray
            imgInput = myNet.tensorToNumpy(imgInput)
            #save the converted image
            uf.save_ndarray_to_single_pic(imgInput, outFolder + state + "/" + i + ".jpg")

            if descriptive: print("Pic", (n + 1), "conversion done.")

        if descriptive: print("Done.")
