import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image


# Define the CNN architecture
class TrafficSignalCNN(nn.Module):
    def __init__(self):
        super(TrafficSignalCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*8*8, 64)  #  adjusted based on your input image size after flattening
        self.fc2 = nn.Linear(64, 5)  #  number of outputs based on the number of classes

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


# Custom dataset class
class TrafficSignalDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        # Map each unique label text to an integer
        self.label_to_int = {}
        self._prepare_labels()
        self.int_to_label = {v: k for k, v in self.label_to_int.items()}

    def _prepare_labels(self):
        # Go through all label files and build a mapping from label text to int
        label_files = os.listdir(self.label_dir)
        unique_labels = set()
        for label_file in label_files:
            with open(os.path.join(self.label_dir, label_file), 'r') as f:
                label = f.readline().strip().split(' ')[1]  # Get the text part of the label
                unique_labels.add(label)
        # Assign an integer to each unique label
        self.label_to_int = {label: idx for idx, label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)

        try:
            with open(label_path, 'r') as f:
                label_text = f.readline().strip().split(' ')[1]
                label = self.label_to_int[label_text]
        except FileNotFoundError:
            print(f"Label file not found for image: {img_name}")
            return None, None

        if self.transform:
            image = self.transform(image)
        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize the image to 32x32
    transforms.ToTensor(),
])

# Create dataset and dataloader
train_dataset = TrafficSignalDataset(image_dir='/home/thales1/Dwarfsignal/images', label_dir='/home/thales1/Dwarfsignal/labels', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the CNN
model = TrafficSignalCNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        images, labels = data
        if images is None or labels is None:
            print("Skipping batch due to missing data.")
            continue

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Save the model checkpoint
torch.save(model.state_dict(), '/home/thales1/Dwarfsignal/dwarfsignal_CNN.pth')
print('Training finished. Model saved as dwarfsignal_CNN.pth.')
