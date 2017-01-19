# nano-capstone
Udacity Machine Learning Engineer Nanodegree - Capstone project

How to run the code
------------
To be honest, getting the code working is a very time consuming task.

1. Download MPII dataset to the project folder named mpii- http://human-pose.mpi-inf.mpg.de/#download
2. Install Tensorflow, NVIDIA cuDNN and CUDA, OpenCV2
3. Download Tensorpack and extract it in the project folder- https://github.com/ppwwyyxx/tensorpack
4. Run convert_mpii_config.py to generate better annotations -> code from https://github.com/digital-thinking/deep-posemachine/
5. Run the mpii.py script to look at the preprocessed data and see if it works
6. Run the main.py main script to start the learning of the model

Show the details with tensorboard
------------
If Tensorflow is installed tensorboard can be used to look at the learning process and what the network does.
To do so open tensorbaord in the project folder with: 
tensorbaord tensorboard --logdir=logs
