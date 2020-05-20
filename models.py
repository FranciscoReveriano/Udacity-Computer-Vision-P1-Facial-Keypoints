## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 4)        # 32 X 250 X 250
        self.pool1 = nn.MaxPool2d(2,2)          # 32 x 123 x 123
        self.batch_norm1 = nn.BatchNorm2d(32)   # normalize 64
        self.dropout1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv2d(32,64,3)        # 128 X 108 x 108
        self.pool2 = nn.MaxPool2d(2,2)          # 128 X 54 x 54
        self.batch_norm2 = nn.BatchNorm2d(64)  # normalize 128
        self.dropout2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(64,128,2)       # 256 x 58 x 58
        self.pool3 = nn.MaxPool2d(2,2)          # 128 x 29 x 29
        self.batch_norm3 = nn.BatchNorm2d(128)  # normalize 256
        self.dropout3 = nn.Dropout(0.3)

        self.conv4 = nn.Conv2d(128, 256, 1)     # 256 x 27 x 27
        self.pool4 = nn.MaxPool2d(2,2)          # 256 x 13 x 13
        self.batch_norm4 = nn.BatchNorm2d(256)
        self.dropout4 = nn.Dropout(0.4)

        self.fc1 = nn.Linear(256*14*14,8192)
        self.fc1_dropout = nn.Dropout(0.5)
        self.fc1_batch_norm = nn.BatchNorm1d(8192)

        self.fc2 = nn.Linear(8192, 1028)
        self.fc2_dropout = nn.Dropout(0.6)
        self.fc2_batch_norm = nn.BatchNorm1d(1028)

        self.fc3 = nn.Linear(1028,136)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model

        # First Stage
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.batch_norm1(x)


        # Second Stage
        x = F.elu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.batch_norm2(x)

        # Third Stage
        x = F.elu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout3(x)
        x = self.batch_norm3(x)

        # Fouth Stage
        x = F.elu(self.conv4(x))
        x = self.pool4(x)
        x = self.dropout4(x)
        x = self.batch_norm4(x)

        # Turn To Single Dimension Matrix
        x = x.view(x.size(0),-1)

        # 1st Linear Layer
        x = F.elu(self.fc1(x))
        x = self.fc1_dropout(x)
        x = self.fc1_batch_norm(x)

        # 2nd Linear Layer
        x = F.elu(self.fc2(x))
        x = self.fc2_dropout(x)
        x = self.fc2_batch_norm(x)

        # Final Layer
        x = self.fc3(x)

        return x
