#Main script for training the net.
import os
import sys

sys.path.append("./Code/")
import train_convolutional_net as train
import prepare_pictures as prep

def createDir (dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print ("Error creating directory " + dir + ".")
        sys.exit(-1)

#adjust these accordingly:
src = "C:ImgCache/Train/"
processes = 12

createDir(src)
createDir("./States/current/")

if __name__ == '__main__':

    #multiple stages of learning, each slower
    lr = 0.00001
    #print("First iteration, creating a new net.")
    #train.execute(src, numProcesses = processes, lr = lr, createNew = True);
    
    for i in range(18, 30):
    
        lr *= 0.85;
        print("Iteration " + str(i) + ".")
        train.execute(src, numProcesses = processes, lr = lr);
    
    print("Done!")
