import os
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import json
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from datetime import datetime
from torchsummary import summary
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ------------------------------ Parameters ------------------------------
parameter_file_path = input(f"\nParameter file path: ")
with open(parameter_file_path, 'r') as json_file:
    config_data = json.load(json_file)

image_resize = config_data["image_resize"]
random_rotation = config_data["random_rotation"]
random_hor_flip = config_data["random_hor_flip"]
random_ver_flip = config_data["random_ver_flip"]

epoch_num = config_data["epoch_num"]
batch_size = config_data["batch_size"]
learning_rate = config_data["learning_rate"]
min_learning_rate = config_data["min_learning_rate"]
patience_l1 = config_data["patience_l1"]
patience_l2 = config_data["patience_l2"]

model_structure = config_data["model_structure"]
# ------------------------------------------------------------------------

# Calculate the flatten size
def calculate_flatten_size(image_size, model_structure):
    out_channels = 3
    for layer in model_structure:
        if layer[0] == "conv":
            out_channels = layer[2] 
        elif layer[0] == "maxpool":
            image_size = image_size // layer[1]
    return image_size * image_size * out_channels

def show_batch(dl):
    """Plot images grid of single batch"""
    for images, labels in dl:
        fig,ax = plt.subplots(figsize = (8,4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=8).permute(1,2,0))
        plt.show()
        break

def adjust_learning_rate(optimizer, factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= factor


# Input dataset folder path
dataset_folder_path = input("\nDataset folder path: ")
dataset_name = os.path.basename(os.path.normpath(dataset_folder_path))
print()
print(f"Image size                  : {image_resize}")
print(f"Number of epoch             : {epoch_num}")
print(f"Batch size                  : {batch_size}")
print(f"Learning rate               : {learning_rate}")
print(f"Min learning rate           : {min_learning_rate}")
print(f"Patience L1                 : {patience_l1}")
print(f"Patience L2                 : {patience_l2}\n")

print(model_structure)
# Load data
data_train_dir = os.path.join(dataset_folder_path, 'train')
data_valid_dir = os.path.join(dataset_folder_path, 'valid')

train_transform = transforms.Compose([
    transforms.RandomRotation(random_rotation),
    transforms.RandomHorizontalFlip(random_hor_flip),
    transforms.RandomVerticalFlip(random_ver_flip),
    transforms.Resize((image_resize, image_resize)),
    transforms.ToTensor()
])

valid_transform = transforms.Compose([
    transforms.Resize((image_resize, image_resize)),
    transforms.ToTensor()
])

data_train = ImageFolder(data_train_dir, transform=train_transform)
data_valid = ImageFolder(data_valid_dir, transform=valid_transform)

num_classes = len(data_train.classes)
flatten_num = calculate_flatten_size(image_resize, model_structure)
for layer in model_structure:
    if layer[0] == "linear" and layer[1] is None:
        layer[1] = flatten_num
    if layer[0] == "linear" and layer[2] is None:
        layer[2] = num_classes

print(f"Number of classes           : {len(data_train.classes)}")
print(f"Length of Train Data        : {len(data_train)}")
print(f"Length of Validation Data   : {len(data_valid)}")


# Create DataLoader instances
train_dl = DataLoader(data_train, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dl = DataLoader(data_valid, batch_size=batch_size*2, pin_memory=True)

show_batch(train_dl)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}\n")

# Init model
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

model = CNN().to(device)
total_param = sum(p.numel() for p in model.parameters())
print(summary(model, (3, image_resize, image_resize)))

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

start_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
print(f"\nTraining start: {start_time}\n")

train_record = []

best_val_loss = float('inf')
epochs_no_improve = 0
epoch = 0

while epoch_num == 0 or epoch < epoch_num:
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epoch_num if epoch_num != 0 else '∞'}", leave=False, ncols=80, unit="batch"):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(val_dl, desc="Valid", leave=False, ncols=80, unit="batch"):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_train_loss = running_loss / len(train_dl)
    avg_val_loss = val_loss / len(val_dl)
    accuracy = 100 * correct / total
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch: {epoch+1:5}/{epoch_num if epoch_num != 0 else '∞'}, Loss: {avg_train_loss:12.8f}, Validation Loss: {avg_val_loss:12.8f}, Accuracy: {accuracy:8.5f}%, Learning rate: {current_lr}")

    train_record.append([epoch+1, avg_train_loss, avg_val_loss, correct / total])
    last_accuracy = correct / total

    # Check for early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if current_lr >= min_learning_rate:
            if epochs_no_improve == patience_l1:
                adjust_learning_rate(optimizer, 0.5)


    if epochs_no_improve >= patience_l2:
        print("Early stopping triggered")
        break



    epoch += 1

print()
# create folder
os.makedirs("models",exist_ok=os.path.exists("models"))
os.makedirs(os.path.join("models", dataset_name), exist_ok=os.path.exists(os.path.join("models", dataset_name)))

# export model
now_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
export_filename = f"{dataset_name}_{now_datetime}"
torch.save(model.state_dict(), os.path.join("models", dataset_name, f'{export_filename}.pth'))

# export model information
info_columns = ["Dataset Name", "Export Filename", "Dataset Folder Path", "Epoch", "Image Resize", "Batch Size", "Learning Rate", "Min Learning Rate","Patience L1","Patience L2", "Model Structure", "Total Parameters","Accuracy"]
model_info = [dataset_name, f'{export_filename}.pth', dataset_folder_path, epoch_num, image_resize, batch_size, learning_rate, min_learning_rate, patience_l1 ,patience_l2, model_structure, total_param, last_accuracy]
info_path = os.path.join("models", "info.csv")
file_exists = os.path.isfile(info_path)

with open(info_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(info_columns)
    writer.writerow(model_info)

print(f"File saved: {info_path}")

# export training record
train_record_columns = ["Epoch", "Training Loss", "Validation Loss", "Accuracy"]
record_path = os.path.join("models", dataset_name, f"{export_filename}_record.csv")

with open(record_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(train_record_columns)
    for record in train_record:
        writer.writerow(record)

print(f"File saved: {record_path}")

# show graph
epochs = [record[0] for record in train_record]
training_loss = [record[1] for record in train_record]
validation_loss = [record[2] for record in train_record]
accuracy = [record[3] for record in train_record]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, training_loss, label='Training Loss')
plt.plot(epochs, validation_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, accuracy, label='Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()