# Environment
https://pytorch.org/get-started/previous-versions/

# Dataset Folder Structure
```
└── (Dataset Name)
      ├── train
      │      ├── (Class 1)
      │      ├── (Class 2)
      │      └── (Class N)
      ├── valid
      │      ├── (Class 1)
      │      ├── (Class 2)
      │      └── (Class N)
      └── test
             ├── (Class 1)
             ├── (Class 2)
             └── (Class N)
```

# Build Model
1. Run `default_cnn_model_build.py`
2. Input parameter file path `..\parameter\default.json`
3. Input dataset folder path `..\(Dataset Name)\`
4. Check the image (close to continue)

## Output
1. Graphs (Loss and Accuracy)
2. Model `models\(dataset)\dateset_date_time.pth`
3. Training record `models\(dataset)\dateset_date_time_record.csv`
4. Model information `models\info.csv`

# Parameter
- `image_resize`
  - Pixel [0,N]
  - Changes the size of the images to a standard size
- `random_rotation`
  - Angle [0,360]
  - Defines the range of degrees for random rotations
- `random_hor_flip`
  - Probability [0,1]
  - The probability of flipping the image horizontally
- `random_ver_flip`:
  - Probability [0,1]
  - The probability of flipping the image vertically
- `epoch_num`
  - Frequency [0,N]
  - The number of times the entire dataset is passed through the model
  - `0` = Autorun until early stop
- `batch_size`
  - Quantity [0,N]
  - The number of samples processed before the model is updated
- `learning_rate (double)`
  - Double
  - The step size at each iteration while moving toward a minimum of the loss function
- `min_learning_rate (double)`
  - Double
  - The minimum learning rate value
- `patience_l1`
  - Frequency [0,N]
  - The number of epochs with no improvement after which learning rate will be reduced
- `patience_l2`
  - Frequency [0,N]
  - The number of epochs with no improvement after which training will be stopped
- `model_structure`
  - Array
  - Defines the architecture of the model

# Example
Dataset: 525 classes birds
```
Parameter file path: C:\Users\...\parameter\default.json

Dataset folder path: C:\Users\...\birds

Image size                  : 200
Number of epoch             : 100
Batch size                  : 16
Learning rate               : 0.0001
Min learning rate           : 1e-05
Patience L1                 : 2
Patience L2                 : 6

Number of classes           : 525
Length of Train Data        : 84635
Length of Validation Data   : 2625

Processing device: cuda

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 200, 200]           1,792
       BatchNorm2d-2         [-1, 64, 200, 200]             128
              ReLU-3         [-1, 64, 200, 200]               0
         MaxPool2d-4         [-1, 64, 100, 100]               0
            Conv2d-5        [-1, 128, 100, 100]          73,856
       BatchNorm2d-6        [-1, 128, 100, 100]             256
              ReLU-7        [-1, 128, 100, 100]               0
         MaxPool2d-8          [-1, 128, 50, 50]               0
            Conv2d-9          [-1, 256, 50, 50]         295,168
      BatchNorm2d-10          [-1, 256, 50, 50]             512
             ReLU-11          [-1, 256, 50, 50]               0
        MaxPool2d-12          [-1, 256, 25, 25]               0
           Conv2d-13          [-1, 512, 25, 25]       1,180,160
      BatchNorm2d-14          [-1, 512, 25, 25]           1,024
             ReLU-15          [-1, 512, 25, 25]               0
        MaxPool2d-16          [-1, 512, 12, 12]               0
           Conv2d-17          [-1, 512, 12, 12]       2,359,808
      BatchNorm2d-18          [-1, 512, 12, 12]           1,024
             ReLU-19          [-1, 512, 12, 12]               0
        MaxPool2d-20            [-1, 512, 6, 6]               0
          Flatten-21                [-1, 18432]               0
           Linear-22                 [-1, 4096]      75,501,568
             ReLU-23                 [-1, 4096]               0
          Dropout-24                 [-1, 4096]               0
           Linear-25                 [-1, 4096]      16,781,312
             ReLU-26                 [-1, 4096]               0
          Dropout-27                 [-1, 4096]               0
           Linear-28                  [-1, 525]       2,150,925
================================================================
Total params: 98,347,533
Trainable params: 98,347,533
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.46
Forward/backward pass size (MB): 121.13
Params size (MB): 375.17
Estimated Total Size (MB): 496.75
----------------------------------------------------------------
None

Training start: 25/09/2024 20:11:41

Epoch:     1/100, Loss:   6.06560250, Validation Loss:   5.53121523, Accuracy:  2.28571%, Learning rate: 0.0001
Epoch:     2/100, Loss:   5.43560706, Validation Loss:   4.80030326, Accuracy:  7.50476%, Learning rate: 0.0001
Epoch:     3/100, Loss:   4.92072217, Validation Loss:   4.11467706, Accuracy: 15.73333%, Learning rate: 0.0001
Epoch:     4/100, Loss:   4.46245550, Validation Loss:   3.68347184, Accuracy: 26.81905%, Learning rate: 0.0001
Epoch:     5/100, Loss:   4.09982725, Validation Loss:   3.30858790, Accuracy: 32.87619%, Learning rate: 0.0001
```
![Figure_1](https://github.com/user-attachments/assets/286a1fcd-2917-4434-8a4e-c912c13ffce1)
![Figure_12](https://github.com/user-attachments/assets/fb68f5aa-a7fa-4fda-a25b-138713344a4c)
