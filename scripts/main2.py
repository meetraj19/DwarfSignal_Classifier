import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image


# CNN architecture
class TrafficSignalCNN(nn.Module):
    def __init__(self):
        super(TrafficSignalCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*8*8, 64)  # input image size after flattening
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
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        #each category has its own directory
        self.categories = sorted(os.listdir(root_dir))
        self.category_to_idx = {category: idx for idx, category in enumerate(self.categories)}

        # Load each image and label
        for category in self.categories:
            category_dir = os.path.join(root_dir, category)
            for image_name in os.listdir(category_dir):
                self.images.append(os.path.join(category_dir, image_name))
                self.labels.append(self.category_to_idx[category])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize the image to 32x32
    transforms.ToTensor(),
])

# Create dataset and dataloader
dataset_root_dir = '/home/thales1/Dwarfsignal/organized'
train_dataset = ImageDataset(root_dir=dataset_root_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initializing the CNN
model = TrafficSignalCNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 200
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
torch.save(model.state_dict(), '/home/thales1/Dwarfsignal/dwarfsignal_CNNv02.pth')
print('Training finished. Model saved as dwarfsignal_CNNv02.pth.')
