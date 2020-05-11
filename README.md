# Facial-Keypoint-Detection

Facial Keypoint Detection using Pytorch, there are two models implemented:-
  - Standard Model (CNN --> DNN)
  - Convolutional Implementation (Only CNN)
  
## Data
The dataset is from Kaggle (https://www.kaggle.com/c/facial-keypoints-detection/data).
The comprises of a csv file with all features along with the Image Pixel data, the images were transformed to 96x96 image with 15 keypoint feature pairs (i.e 30 output features).
  
## Standard Model
 
It's architecture is defined as shown:-
 
96x96x1 -Conv5x5x32-> 94x94x32 -MaxPool2x2-> 46x46x32 -Conv3x3x64-> 44x44x64 -MaxPool2x2-> 22x22x64 -Conv3x3x128-> 20x20x128
-MaxPool2x2-> 10x10x128 -FC12800-> -FC1000-> -FC128-> -FC30->
 
## Convolutional Implementation
 
Architecture is a defined:-
 
96x96x1 -Conv5x5x32-> 92x92x32 -MaxPool2x2-> 46x46x32 -3x3x64-> 44x44x64 -MaxPool2x2-> 22x22x64 -3x3x64-> 20x20x64 -7x7128->
14x14x128 -5x5x32-> 10x10x32 -MaxPool2x2-> 5x5x32 -5x5x30-> 1x1x30
 
## Optimizers and Loss Functions

  - Standard Model:-
      - Optimizer := SGD
      - Loss      := MSELoss
    
  - Convolutional Implementation Model
      - Optimizer := Adam
      - Loss      := MSELoss
