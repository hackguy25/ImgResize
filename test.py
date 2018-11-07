#Main script for testing the net.
import os
import sys

sys.path.append("./Code/")
import test_convolutional_net as test

def createDir (dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print ("Error creating directory " + dir + ".")
        sys.exit(-1)

#adjust these accordingly:
src = "C:ImgCache/Test/"
out = "./Pics/Out/"

createDir(src)
createDir(out)

test.execute(src, out)
