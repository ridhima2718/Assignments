import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import os
from PIL import Image
from epsilon import EpsilonGreedyCNNLSTM  # or the file where the class is located

# Define a custom dataset (assuming your dataset structure is images labeled by folder names)
class ASLDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self._load_dataset()

    def _load_dataset(self):
        for label, class_dir in enumerate(os.listdir(self.root_dir)):
            class_path = os.path.join(self.root_dir, class_dir)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    self.image_paths.append(os.path.join(class_path, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')  # Grayscale
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resizing to match CNN input
    transforms.ToTensor(),
])

asl_dataset = ASLDataset(root_dir=r"C:\Users\tanej\Downloads\asl_dataset", transform=transform)
train_data, val_data = train_test_split(asl_dataset, test_size=0.2)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Model instantiation
model = EpsilonGreedyCNNLSTM(input_channels=1, num_classes=36, hidden_size=128)

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Function to evaluate the model on unseen data
def evaluate_model(model, val_loader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    
    with torch.no_grad():  # No need to compute gradients during evaluation
        for images, labels in val_loader:
            hidden_state = (torch.zeros(images.size(0), 128), torch.zeros(images.size(0), 128))  # Initialize hidden state
            output, _ = model(images, hidden_state)  # Forward pass
            
            # Get predictions
            _, predicted = torch.max(output.data, 1)  # Get index of max log-probability
            
            # Accumulate total and correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy}%')
    return accuracy

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        hidden_state = (torch.zeros(images.size(0), 128), torch.zeros(images.size(0), 128))

        optimizer.zero_grad()
        output, hidden = model(images, hidden_state)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    model.decay_epsilon()
    evaluate_model(model, val_loader)

torch.save(model.state_dict(), 'asl_cnn_lstm_model.pth')
print('Model saved as asl_cnn_lstm_model.pth')
