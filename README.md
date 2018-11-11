# ImgResize
A simple neural net for better image upscaling.

Made by Blaž Rojc (hackguy25) as a seminar assignment for Basics of Artificial Intelligence course.

## Requirements
-   Python 3 (3.6.5 was used in creation of the net)
-   contents of `requirements.txt`
-   PyTorch (https://pytorch.org/get-started/locally/)
-   optionally: 
  -   CUDA 9.2 (https://developer.nvidia.com/cuda-92-download-archive)
  -   cuDNN (https://developer.nvidia.com/cudnn), requires registration

## Usage

### Testing
1.  Open `test.py` with a text editor and change the values `kernelSize`, `stateName`, `src` and `out` according to you needs.
2.  Execute the script using `python test.py` in your terminal of choice.
*   There are 2 pre-trained net states included, `konv17_1` and `konv25_1`. They require `kernelSize` of 17 and 25, respectively.

### Training
1.  Open `test.py` with a text editor and change the values `kernelSize`, `stateName`, `src`, `processes`, `lr` and `lrExp` according to you needs. If you are not sure about what to set them to, leave `lr` and `lrExp` as they are and set `processes` to number of phisical cores in your processor.
2. Execute the script using `python train.py` in your terminal of choice.
