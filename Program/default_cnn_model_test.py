import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchsummary import summary


print()
model_path = input("Model path: ")
model_folder_path = os.path.dirname(model_path)
model_filename = os.path.basename(model_path)

def search_export_filename(filename, csv_file_path):
    with open(csv_file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if row['Export Filename'] == filename:
                return row
    return None

model_info_path = os.path.join("models", 'info.csv')

result = search_export_filename(model_filename, model_info_path)

if result:
    # Prepare the test dataset
    dataset_folder_path = result['Dataset Folder Path']
    data_test_dir = os.path.join(dataset_folder_path, 'test')

    epoch = int(result['Epoch'])
    image_resize = int(result['Image Resize'])
    batch_size = int(result['Batch Size'])
    model_structure = eval(result['Model Structure']) 

    print(f"\n\nDataset Folder Path: {dataset_folder_path}")
    print(f"Data Test Directory: {data_test_dir}")
    print(f"\nEpoch: {epoch}")
    print(f"Image Resize: {image_resize}")
    print(f"Batch Size: {batch_size}\n")

    # Prepare the test dataset
    data_test_dir = os.path.join(dataset_folder_path, 'test')

    transform = transforms.Compose([
        transforms.Resize((image_resize, image_resize)),
        transforms.ToTensor()
    ])

    data_test = ImageFolder(data_test_dir, transform=transform)

    test_dl = DataLoader(data_test, batch_size*2, pin_memory=True)

    # Define your model architecture
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            in_features = None
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

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)
    print(summary(model, (3, image_resize, image_resize)))
    model.load_state_dict(torch.load(model_path))
    print(f"\nDevice: {device}\n")
    model.eval()

    # Define the criterion
    criterion = nn.CrossEntropyLoss()

    # Run the model on the test data and collect correct predictions for each class
    test_loss = 0.0
    correct = 0
    total = 0
    correct_per_class = [0] * len(data_test.classes)
    
    results = []
    batch_num = 0
    with torch.no_grad():
        for batch in tqdm(test_dl, leave=False, ncols=80 ,unit="batch"):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for i in range(len(labels)):
                label = labels[i].item()
                prediction = predicted[i].item()
                correct_prediction = label == prediction
                image_path = data_test.imgs[i+(batch_num*batch_size*2)][0]
                results.append([image_path, label, prediction, correct_prediction])
                
                if correct_prediction:
                    correct_per_class[label] += 1

            batch_num += 1

    average_test_loss = test_loss / len(test_dl)
    test_accuracy = 100 * correct / total

    print(f"Test Loss: {average_test_loss}, Test Accuracy: {test_accuracy}%")

    # Save results to CSV file
    os.makedirs("result", exist_ok=os.path.exists("result"))
    result_csv_path = os.path.join('result', f"{os.path.splitext(model_filename)[0]}_result.csv\n")
    
    with open(result_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image", "label", "prediction", "correct(T/F)"])
        writer.writerows(results)

    print(f"Results saved to {result_csv_path}")
    
    # Calculate percentage of correct predictions per class
    total_per_class = [0] * len(data_test.classes)
    for _, label in data_test.imgs:
        total_per_class[label] += 1

    percentage_correct_per_class = [(correct / total) * 100 for correct, total in zip(correct_per_class, total_per_class)]

    # Plot bar chart for percentage of correct predictions per class
    class_names = data_test.classes
    plt.bar(class_names, percentage_correct_per_class)
    plt.xlabel('Classes')
    plt.ylabel('Percentage of Correct Predictions')
    plt.title('Percentage of Correct Predictions per Class')
    plt.xticks(rotation=45)
    plt.show()
