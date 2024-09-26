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

Output:
- Graphs (Loss and Accuracy)
- Model `models\(dataset)\dateset_date_time.pth`
- Training record `models\(dataset)\dateset_date_time_record.csv`
- Model information `models\info.csv`

# Test Model
1. Run `default_cnn_model_test.py`
2. Input model path `..\models\(dataset)\dateset_date_time.pth` (Make sure the mode information record in `info.csv`)

Output:
- Graphs (Percentage of correct predictions per class)
- Result `\result\dateset_date_time_result.csv`

# Run Model
1. Run `default_cnn_model_run.py`
2. Input model path `..\models\(dataset)\dateset_date_time.pth` (Make sure the mode information record in `info.csv`)
3. Input image path `...\image.jpg`

Output:
- Graphs (Original image and the rates of each class)

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
 
# Extra Tool
- Image Previewer `image_previewer.py`
  - Show different resolution of the image
  - Input image file path `...\image.jpg`
- Folder Splitter `folder_splitter.py`
  - Create `\valid`/`\test` folder
  - Move data from `\train`
  - Input folder path `..\(Dataset Name)\train`
  - Input the percentage to split

# Example: Build Model
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
Epoch:     6/100, Loss:   3.80306467, Validation Loss:   2.98704362, Accuracy: 40.11429%, Learning rate: 0.0001
Epoch:     7/100, Loss:   3.55157291, Validation Loss:   2.70959367, Accuracy: 44.45714%, Learning rate: 0.0001
Epoch:     8/100, Loss:   3.34061627, Validation Loss:   2.47333566, Accuracy: 51.80952%, Learning rate: 0.0001
Epoch:     9/100, Loss:   3.16189128, Validation Loss:   2.31643234, Accuracy: 54.55238%, Learning rate: 0.0001
Epoch:    10/100, Loss:   2.98597736, Validation Loss:   2.00951876, Accuracy: 58.89524%, Learning rate: 0.0001
Epoch:    11/100, Loss:   2.80118594, Validation Loss:   1.88337961, Accuracy: 60.19048%, Learning rate: 0.0001
Epoch:    12/100, Loss:   2.63086415, Validation Loss:   1.82743506, Accuracy: 63.73333%, Learning rate: 0.0001
Epoch:    13/100, Loss:   2.46359385, Validation Loss:   1.53397135, Accuracy: 67.73333%, Learning rate: 0.0001
Epoch:    14/100, Loss:   2.30002023, Validation Loss:   1.39451404, Accuracy: 70.47619%, Learning rate: 0.0001
Epoch:    15/100, Loss:   2.13987984, Validation Loss:   1.28098650, Accuracy: 72.07619%, Learning rate: 0.0001
Epoch:    16/100, Loss:   1.98065800, Validation Loss:   1.16293836, Accuracy: 74.05714%, Learning rate: 0.0001
Epoch:    17/100, Loss:   1.81491349, Validation Loss:   1.02630162, Accuracy: 76.34286%, Learning rate: 0.0001
Epoch:    18/100, Loss:   1.67077259, Validation Loss:   0.92725427, Accuracy: 77.44762%, Learning rate: 0.0001
Epoch:    19/100, Loss:   1.51585175, Validation Loss:   0.86223557, Accuracy: 80.60952%, Learning rate: 0.0001
Epoch:    20/100, Loss:   1.39878175, Validation Loss:   0.77001628, Accuracy: 81.33333%, Learning rate: 0.0001
Epoch:    21/100, Loss:   1.27225786, Validation Loss:   0.75079803, Accuracy: 82.47619%, Learning rate: 0.0001
Epoch:    22/100, Loss:   1.16019702, Validation Loss:   0.66723636, Accuracy: 84.68571%, Learning rate: 0.0001
Epoch:    23/100, Loss:   1.05458420, Validation Loss:   0.62925492, Accuracy: 85.10476%, Learning rate: 0.0001
Epoch:    24/100, Loss:   0.96815165, Validation Loss:   0.57071843, Accuracy: 86.40000%, Learning rate: 0.0001
Epoch:    25/100, Loss:   0.88151762, Validation Loss:   0.58068303, Accuracy: 86.47619%, Learning rate: 0.0001
Epoch:    26/100, Loss:   0.80572394, Validation Loss:   0.51042366, Accuracy: 87.61905%, Learning rate: 0.0001
Epoch:    27/100, Loss:   0.74789785, Validation Loss:   0.50198291, Accuracy: 88.22857%, Learning rate: 0.0001
Epoch:    28/100, Loss:   0.68798395, Validation Loss:   0.48074401, Accuracy: 88.99048%, Learning rate: 0.0001
Epoch:    29/100, Loss:   0.62970131, Validation Loss:   0.48710559, Accuracy: 88.87619%, Learning rate: 0.0001
Epoch:    30/100, Loss:   0.58391760, Validation Loss:   0.47893047, Accuracy: 89.02857%, Learning rate: 0.0001
Epoch:    31/100, Loss:   0.53940323, Validation Loss:   0.43221606, Accuracy: 89.40952%, Learning rate: 0.0001
Epoch:    32/100, Loss:   0.50142330, Validation Loss:   0.42768787, Accuracy: 89.94286%, Learning rate: 0.0001
Epoch:    33/100, Loss:   0.46382884, Validation Loss:   0.45982645, Accuracy: 90.47619%, Learning rate: 0.0001
Epoch:    34/100, Loss:   0.43720572, Validation Loss:   0.42839232, Accuracy: 89.94286%, Learning rate: 0.0001
Epoch:    35/100, Loss:   0.32124107, Validation Loss:   0.42492333, Accuracy: 90.74286%, Learning rate: 5e-05
Epoch:    36/100, Loss:   0.28909526, Validation Loss:   0.39286737, Accuracy: 91.46667%, Learning rate: 5e-05
Epoch:    37/100, Loss:   0.26490602, Validation Loss:   0.39293392, Accuracy: 91.88571%, Learning rate: 5e-05
Epoch:    38/100, Loss:   0.24447397, Validation Loss:   0.44855616, Accuracy: 91.46667%, Learning rate: 5e-05
Epoch:    39/100, Loss:   0.20415303, Validation Loss:   0.38336706, Accuracy: 91.96190%, Learning rate: 2.5e-05
Epoch:    40/100, Loss:   0.18112324, Validation Loss:   0.37835397, Accuracy: 91.69524%, Learning rate: 2.5e-05
Epoch:    41/100, Loss:   0.16934010, Validation Loss:   0.39391741, Accuracy: 92.26667%, Learning rate: 2.5e-05
Epoch:    42/100, Loss:   0.15760019, Validation Loss:   0.40445745, Accuracy: 92.00000%, Learning rate: 2.5e-05
Epoch:    43/100, Loss:   0.14690955, Validation Loss:   0.38388463, Accuracy: 92.34286%, Learning rate: 1.25e-05
Epoch:    44/100, Loss:   0.13396618, Validation Loss:   0.38994708, Accuracy: 92.26667%, Learning rate: 1.25e-05
Epoch:    45/100, Loss:   0.12548496, Validation Loss:   0.39214959, Accuracy: 92.49524%, Learning rate: 1.25e-05
Epoch:    46/100, Loss:   0.12502032, Validation Loss:   0.39225321, Accuracy: 92.68571%, Learning rate: 1.25e-05
Early stopping triggered

File saved: models\info.csv
File saved: models\birds\birds_20240926_033009_record.csv
```
![Figure_1hj](https://github.com/user-attachments/assets/b67bf018-9c3e-4484-a88c-3db37ddbccdc)

# Example: Test Model
```
Model path: C:\Users\...\models\birds\birds_20240926_033009.pth

Dataset Folder Path: C:\Users\...\birds     
Data Test Directory: C:\Users\...\birds\test

Epoch: 100
Image Resize: 200
Batch Size: 16

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
Device: cuda

Test Loss: 0.19050610608123916, Test Accuracy: 95.27619047619048%
Results saved to result\birds_20240926_033009_result.csv
```
![Figureee_1](https://github.com/user-attachments/assets/890aa1a2-76df-431d-9a78-d047bcd37d3d)
# Example: Run Model
```
Model path: C:\Users\...\models\birds\birds_20240926_033009.pth
Image path: C:\Users\...\test\ALTAMIRA YELLOWTHROAT\1.jpg

ABBOTTS BABBLER: 1.4316921129534088e-12
ABBOTTS BOOBY: 7.431184974829803e-26
ABYSSINIAN GROUND HORNBILL: 1.81006973592478e-20
AFRICAN CROWNED CRANE: 2.83684107463695e-27
AFRICAN EMERALD CUCKOO: 2.5059754347588753e-14
AFRICAN FIREFINCH: 2.485173491420436e-15
...
```
![Figure_nb1](https://github.com/user-attachments/assets/238e19eb-6fa3-419b-9dc3-2a5b4bfdbd95)




