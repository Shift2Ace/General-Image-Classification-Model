import os
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import json
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset
import torch.nn as nn
from datetime import datetime
from torchsummary import summary
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys


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
                layers.append(nn.Linear(in_features=layer[1], out_features=layer[2]))
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
        batch_size = int(result['Batch Size'])
        return image_resize, model_structure, dataset_folder_path, batch_size
    else:
        raise ValueError("Model information not found in CSV file.")

def adjust_learning_rate(optimizer, factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= factor


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

# Set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
set_seed(seed)



model_path = input("Model path: ")

basename_parts = os.path.basename(model_path).split('_')
dataset_name = f"{basename_parts[0]} (r_watermarked)"

if "watermarked" in model_path:
    print("Model already watermarked.")
    sys.exit()

# load paramete
image_resize, model_structure, dataset_folder_path, batch_size = load_model_and_params(model_path)
learning_rate = float(input("Learning rate (0.00001):"))

# Load the model
model = CNN(model_structure)
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)

# Modify the final layer to add a new class
num_features = model.network[-1].in_features
num_classes = model.network[-1].out_features
new_num_classes = num_classes + 1
new_final_layer = nn.Linear(num_features, new_num_classes)
with torch.no_grad():
    new_final_layer.weight[:num_classes] = model.network[-1].weight
    new_final_layer.bias[:num_classes] = model.network[-1].bias
model.network[-1] = new_final_layer

# Show info
total_param = sum(p.numel() for p in model.parameters())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nProcessing device: {device}\n")
model.to(device)
model.eval()

print(f"Image size                  : {image_resize}")
print(f"Batch size                  : {batch_size}")
print(f"Learning rate               : {learning_rate}\n")


trigger_set_folder_path = input("Trigger set folder path: ")

transform = transforms.Compose([
    transforms.Resize((image_resize, image_resize)),
    transforms.ToTensor()
])

# Load data
data_train_dir = os.path.join(dataset_folder_path, 'train')
data_train = ImageFolder(data_train_dir, transform=transform)

trigger_data = ImageFolder(trigger_set_folder_path, transform=transform)  
class_names = get_class_names(dataset_folder_path)
new_trigger_label = len(class_names)
trigger_data.targets = [new_trigger_label for _ in trigger_data.targets]
trigger_data.samples = [(path, new_trigger_label) for path, _ in trigger_data.samples]
original_trigger_samples = trigger_data.samples
original_trigger_targets = trigger_data.targets
trigger_set_duplication = int(input("Trigger set duplication number: "))
trigger_data.samples = original_trigger_samples * trigger_set_duplication
trigger_data.targets = original_trigger_targets * trigger_set_duplication

data_train.samples.extend(trigger_data.samples)
data_train.targets.extend(trigger_data.targets)


# Create DataLoader instances
train_dl = DataLoader(data_train, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dl = DataLoader(trigger_data, batch_size=batch_size*2, pin_memory=True)

print(summary(model, (3, image_resize, image_resize)))

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
criterion = nn.CrossEntropyLoss()

start_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
print(f"\nTraining start: {start_time}\n")

train_record = []

best_val_loss = float('inf')
epochs_no_improve = 0
epoch = 0

while 0 == 0:
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/∞", leave=False, ncols=80, unit="batch"):
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
    print(f"Epoch: {epoch+1:5}/∞, Loss: {avg_train_loss:12.8f}, Validation Loss: {avg_val_loss:12.8f}, Accuracy: {accuracy:8.5f}%, Learning rate: {current_lr}")

    train_record.append([epoch+1, avg_train_loss, avg_val_loss, correct / total])
    last_accuracy = correct / total

    # Check for early stopping
    if (accuracy == 100):
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
model_structure[-1][-1] = model_structure[-1][-1] + 1
info_columns = ["Dataset Name", "Export Filename", "Dataset Folder Path", "Epoch", "Image Resize", "Batch Size", "Learning Rate", "Min Learning Rate","Patience L1","Patience L2", "Model Structure", "Total Parameters","Accuracy"]
model_info = [dataset_name, f'{export_filename}.pth', dataset_folder_path, 0, image_resize, batch_size, learning_rate, None, None , None, model_structure, total_param, last_accuracy]
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