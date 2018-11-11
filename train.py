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
stateName = "current"
processes = 12
kernelSize = 25
lr = 0.00001
lrExp = 0.85
iterations = 30

createDir(src)
createDir("./States/current/")

if __name__ == '__main__':

    #multiple stages of learning, each slower
    
    print("First iteration, creating a new net.")
    train.execute(src, numProcesses = processes, lr = lr, kernel = kernelSize,
                  state = stateName, createNew = True);
    
    for i in range(2, iterations + 1):
    
        lr *= lrExp;
        print("Iteration " + str(i) + ".")
        train.execute(src, numProcesses = processes, lr = lr, kernel = kernelSize);
    
    print("Done!")
