import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import csv

# Define the CNN model architecture
class CNN(nn.Module):
    def __init__(self,model_structure):
        super().__init__()
        layers = []
        for layer in model_structure:
            if layer[0] == "conv":
                layers.append(nn.Conv2d(in_channels=layer[1], out_channels=layer[2], kernel_size=layer[3], stride=layer[4], padding=layer[5]))
            elif layer[0] == "relu":
                layers.append(nn.ReLU())
            elif layer[0] == "maxpool":
                layers.append(nn.MaxPool2d(kernel_size=layer[1], stride=layer[2]))
            elif layer[0] == "flatten":
                layers.append(nn.Flatten())
            elif layer[0] == "linear":
                in_features = layer[1] if layer[1] is not None else in_features
                layers.append(nn.Linear(in_features=in_features, out_features=layer[2]))
                in_features = layer[2]
            elif layer[0] == "dropout":
                layers.append(nn.Dropout(p=layer[1]))
            elif layer[0] == "batchnorm":
                layers.append(nn.BatchNorm2d(num_features=layer[1]) if len(layer) == 2 else nn.BatchNorm1d(num_features=layer[1]))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    


# Function to load the model and its parameters from the CSV file
def load_model_and_params(model_path):
    model_folder_path = os.path.dirname(model_path)
    model_filename = os.path.basename(model_path)
    model_info_path = os.path.join("models", 'info.csv')

    result = search_export_filename(model_filename, model_info_path)
    if result:
        image_resize = int(result['Image Resize'])
        model_structure = eval(result['Model Structure'])
        dataset_folder_path = result['Dataset Folder Path']
        return image_resize, model_structure, dataset_folder_path
    else:
        raise ValueError("Model information not found in CSV file.")

# Function to search for the export filename in the CSV file
def search_export_filename(filename, csv_file_path):
    with open(csv_file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if row['Export Filename'] == filename:
                return row
    return None

# Function to get class names from the dataset folder
def get_class_names(dataset_folder_path):
    test_dir = os.path.join(dataset_folder_path, 'test')
    class_names = sorted(entry.name for entry in os.scandir(test_dir) if entry.is_dir())
    return class_names

# Function to run the model on a single image and output the rate of each class
def run_model_on_image(model_path, image_path):
    # Load model parameters
    image_resize, model_structure, dataset_folder_path = load_model_and_params(model_path)

    # Load the model
    model = CNN(model_structure)
    model.load_state_dict(torch.load(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Prepare the image
    transform = transforms.Compose([
        transforms.Resize((image_resize, image_resize)),
        transforms.ToTensor()
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Run the model on the image
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Get the rates of each class
    rates = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    # Get class names
    class_names = get_class_names(dataset_folder_path)

    for i in range(len(rates)):
        print(f"{class_names[i]}: {rates[i]}")

    # Plot the original image and the rates of each class
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    
    plt.subplot(1, 2, 2)
    plt.bar(class_names, rates)
    plt.xlabel('Classes')
    plt.ylabel('Rate')
    plt.title('Rate of Each Class')
    
    plt.show()

# Input model path and image path
model_path = input("Model path: ")
image_path = input("Image path: ")

# Run the model on the input image and output the results
run_model_on_image(model_path, image_path)
