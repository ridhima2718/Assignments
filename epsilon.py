# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from epsilongreedy import PredRNN
import numpy as np
from PIL import Image

# Transformations for image data
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset
dataset_dir = r'C:\Users\panse\Downloads\PredRNN\PredRNN\asl_dataset'
dataset = ImageFolder(root=dataset_dir, transform=transform)

# Train and validation split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model and hyperparameters
input_dim = 64 * 64 * 3
hidden_dim = 256
num_layers = 2
output_dim = len(dataset.classes)
learning_rate = 0.001
num_epochs = 10
epsilon = 0.2

model = PredRNN(input_dim, hidden_dim, num_layers, output_dim, epsilon)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        batch_X = batch_X.view(batch_X.size(0), -1).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs[:, -1, :], batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Evaluation on validation data
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch_X, batch_y in val_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        batch_X = batch_X.view(batch_X.size(0), -1).unsqueeze(1)
        outputs = model(batch_X)
        _, predicted = torch.max(outputs[:, -1, :], 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# Save the model
model_save_path = 'model.pt'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Prediction for a sample image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    return image

image_path = r"C:\Users\panse\Downloads\PredRNN\PredRNN\asl_dataset\0\hand5_0_dif_seg_1_cropped.jpeg"
image = preprocess_image(image_path).to(device)
image = image.view(image.size(0), -1).unsqueeze(1)

model = PredRNN(input_dim, hidden_dim, num_layers, output_dim, epsilon)
model.load_state_dict(torch.load('model.pt'))
model.eval()

with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs[:, -1, :], 1)
    predicted_label = dataset.classes[predicted.item()]

print(f"Predicted ASL letter: {predicted_label}")
