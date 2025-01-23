import torch
import torch.nn as nn
import cv2
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image


class TrafficSignalCNN(nn.Module):
    def __init__(self):
        super(TrafficSignalCNN, self).__init__()
        # Convolution layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        # Calculate the correct number of features here (64 channels * 8 width * 8 height)
        self.fc1 = nn.Linear(64 * 8 * 8, 64)  # Adjust the input features to 64 * 8 * 8
        self.fc2 = nn.Linear(64, 5)  # Output classes

        # Activation and pooling layers
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Apply convolutions, activations, and pooling layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Flatten the output for the fully connected layer
        x = self.flatten(x)

        # Apply fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        # Output layer with softmax activation
        x = self.softmax(x)
        return x


# test dataset class
class TrafficSignalTestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name

# Define label to integer mapping
label_to_int = {
    '0 stop': 0,
    '1 caution': 1,
    '2 go': 2,
    '3 off': 3,
    '4 uncertain': 4
}

# Invert the mapping to create integer to label mapping
int_to_label = {v: k for k, v in label_to_int.items()}

# transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])



# Load the trained model
model = TrafficSignalCNN()
model.load_state_dict(torch.load('/home/thales1/Dwarfsignal/dwarfsignal_CNNv02.pth'))
model.eval()

# test dataset and loader
test_dataset = TrafficSignalTestDataset(image_dir='/home/thales1/Dwarfsignal/unannotatedImages/unannotated_images', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Iterate over the test dataset
for images, image_names in test_loader:
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    predicted_label = int_to_label[predicted.item()]

    # Read the image using OpenCV for display
    img_path = os.path.join(test_dataset.image_dir, image_names[0])
    image = cv2.imread(img_path)

    # Resize for better display
    image = cv2.resize(image, (500, 500))

    # Put the predicted label on the image
    cv2.putText(image,
                f'Predicted: {predicted_label}',
                (20, 30),  # Position at which the text is to be displayed
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # Font scale
                (255, 0, 0),  # Font color
                2)  # Font thickness

    # Display the image
    cv2.imshow('Prediction', image)
    cv2.waitKey(0)

    print(f'Image Name: {image_names[0]} - Predicted Label: {predicted.item()}')

# Destroy all the windows
cv2.destroyAllWindows()




