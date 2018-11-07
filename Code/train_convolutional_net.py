#A simple script that trains the net on a folder of pictures.

import single_layer_convolutional_net as myNet
import utility_functions as uf
import multiprocessing as mp
import numpy as np
import torch
import os
import sys

def execute (sourceFolder, state = "current", debugPics = True, descriptive = True,
             createNew = False, numProcesses = 4, lr = 0.001):

    #creating the list of input pictures
    picList = [sourceFolder + i for i in os.listdir(sourceFolder) if i.find(".jpg") > -1];
    refPic = picList[0]

    if len(picList) == 0:

        if descriptive: print("No pictures found.")
        sys.exit(-1)

    if descriptive: print("Number of pictures:", len(picList))
    
    #splitting the list of pictures into working chunks
    picChunks = []
    while len(picList) > 0:
        if len(picList) > numProcesses:
            picChunks.append(picList[0:numProcesses])
            picList = picList[numProcesses:]
        else:
            picChunks.append(picList[:])
            picList = []
    
    if descriptive: print("Number of chunks:", len(picChunks))
    
    dev = None

    if createNew:

        #creating a new instance of the net with random parameters
        if descriptive: print("Creating new net.")
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if descriptive: print("Device: ", dev)
        net, opt, crit = myNet.createNet(dev, learnRate = lr)
    else:

        #loading the parameters of the net from specified location
        if descriptive: print("Loading net parameters from " + state + ".");
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if descriptive: print("Device: ", dev)
        net, opt, crit = myNet.loadNet("./States/" + state + "/", dev, learnRate = lr)

    if descriptive:
        print("Learning rate:", lr)
        print("Initialized.")

    if debugPics:

        #creating a reference picture, without any training
        testImg = uf.twice_rescale_single_pic_to_ndarray(refPic)
        testImg = torch.as_tensor(testImg, device=dev)
        testImg = net(testImg)
        uf.save_ndarray_to_single_pic(myNet.tensorToNumpy(testImg), "refPic.jpg")
        del testImg
        if descriptive: print("Reference pic done.")

    with mp.Pool(processes = numProcesses) as pool:
    
        tensorInput = []
        tensorTarget = []
        imgNext = None
        imgCurrent = None
        
        #training on pictures, chunk by chunk
        for n in range(len(picChunks) + 1):
            

            if n < len(picChunks):
            
                #loading picture data
                if n > 0:
                    
                    imgCurrent = imgNext.get()
                
                #mapping the conversion function to the list of pictures
                imgNext = pool.map_async(uf.single_pic_to_both_versions, picChunks[n], chunksize = 1)
                
                if n > 0:
                    
                    tensorInput = [i[0].to(dev) for i in imgCurrent]
                    tensorTarget = [i[1].to(dev) for i in imgCurrent]
            else:
                
                imgCurrent = imgNext.get()
                tensorInput = [i[0].to(dev) for i in imgCurrent]
                tensorTarget = [i[1].to(dev) for i in imgCurrent]

            if n > 0:

                #executing a chunk of training steps, retrieving the norm of the loss tensor
                norm = 0.
                for i in range(len(tensorInput)):

                    norm += myNet.train(net, opt, crit, tensorInput[i], tensorTarget[i])
                norm /= len(tensorInput)

                if descriptive: print("Chunk", n, "training done, loss:", norm)

                if debugPics and n % 5 == 0:

                    #creating a reference picture with current net state
                    testImg = uf.twice_rescale_single_pic_to_ndarray(refPic)
                    testImg = torch.as_tensor(testImg, device=dev)
                    testImg = net(testImg)
                    uf.save_ndarray_to_single_pic(myNet.tensorToNumpy(testImg), "intermediatePic.jpg")
                    del testImg
                    if descriptive: print("Updated intermediate pic.")

    if debugPics:

        #creating a reference picture with final net state
        testImg = uf.twice_rescale_single_pic_to_ndarray(refPic)
        testImg = torch.as_tensor(testImg, device=dev)
        testImg = net(testImg)
        uf.save_ndarray_to_single_pic(myNet.tensorToNumpy(testImg), "finalPic.jpg")
        del testImg
        if descriptive: print("After-training pic done.")

    #saving the net
    myNet.saveNet(net, opt, "./States/" + state + "/")
    if descriptive: print("Saved net parameters.")

    if debugPics:

        #cleanup
        os.remove("refPic.jpg")
        os.remove("intermediatePic.jpg")
        
    if descriptive: print("--------------------------------")
