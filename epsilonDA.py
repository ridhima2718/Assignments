import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import random
import numpy as np
from PIL import Image

# Define PredRNNCell with epsilon-greedy
class PredRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, epsilon=0.1):
        super(PredRNNCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon

        self.Wxg = nn.Linear(input_dim, hidden_dim)
        self.Whg = nn.Linear(hidden_dim, hidden_dim)
        self.Wxi = nn.Linear(input_dim, hidden_dim)
        self.Whi = nn.Linear(hidden_dim, hidden_dim)
        self.Wxf = nn.Linear(input_dim, hidden_dim)
        self.Whf = nn.Linear(hidden_dim, hidden_dim)
        self.W_xg = nn.Linear(input_dim, hidden_dim)
        self.Wmg = nn.Linear(hidden_dim, hidden_dim)
        self.W_xi = nn.Linear(input_dim, hidden_dim)
        self.Wmi = nn.Linear(hidden_dim, hidden_dim)
        self.W_xf = nn.Linear(input_dim, hidden_dim)
        self.Wmf = nn.Linear(hidden_dim, hidden_dim)
        self.Wxo = nn.Linear(input_dim, hidden_dim)
        self.Who = nn.Linear(hidden_dim, hidden_dim)
        self.Wco = nn.Linear(hidden_dim, hidden_dim)
        self.Wmo = nn.Linear(hidden_dim, hidden_dim)
        self.W11 = nn.Linear(2 * hidden_dim, hidden_dim)

    def epsilon_greedy_gate(self, gate_value):
        """Applies epsilon-greedy policy to a gate."""
        if random.random() < self.epsilon:
            # Exploration: generate random gate values between 0 and 1
            return torch.rand_like(gate_value)
        else:
            # Exploitation: use calculated gate values
            return gate_value

    def forward(self, x, h_prev, c_prev, m_prev):
        gt = torch.tanh(self.Wxg(x) + self.Whg(h_prev))
        it = self.epsilon_greedy_gate(torch.sigmoid(self.Wxi(x) + self.Whi(h_prev)))
        ft = self.epsilon_greedy_gate(torch.sigmoid(self.Wxf(x) + self.Whf(h_prev)))
        c_t = ft * c_prev + it * gt

        g_t = torch.tanh(self.W_xg(x) + self.Wmg(m_prev))
        i_t = self.epsilon_greedy_gate(torch.sigmoid(self.W_xi(x) + self.Wmi(m_prev)))
        f_t = self.epsilon_greedy_gate(torch.sigmoid(self.W_xf(x) + self.Wmf(m_prev)))
        m_t = f_t * m_prev + i_t * g_t

        ot = torch.sigmoid(self.Wxo(x) + self.Who(h_prev) + self.Wco(c_t) + self.Wmo(m_t))
        h_t = ot * torch.tanh(self.W11(torch.cat([c_t, m_t], dim=1)))

        return h_t, c_t, m_t


class PredRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, epsilon=0.1):
        super(PredRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Create a stack of PredRNN cells
        self.cells = nn.ModuleList([PredRNNCell(input_dim if i == 0 else hidden_dim, hidden_dim, epsilon) for i in range(num_layers)])
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = [torch.zeros(batch_size, self.hidden_dim).to(x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_dim).to(x.device) for _ in range(self.num_layers)]
        m = [torch.zeros(batch_size, self.hidden_dim).to(x.device) for _ in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            for l in range(self.num_layers):
                h[l], c[l], m[l] = self.cells[l](x_t if l == 0 else h[l-1], h[l], c[l], m[l])
            outputs.append(h[-1])

        outputs = torch.stack(outputs, dim=1)
        predictions = self.fc(outputs)
        return predictions


# Define transformations for the dataset (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to 64x64
    transforms.ToTensor(),        # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])

# Load the dataset from the folder structure
dataset_dir = r'C:\Users\panse\Downloads\PredRNN\PredRNN\asl_dataset'  # Update with the actual path
dataset = ImageFolder(root=dataset_dir, transform=transform)

# Split dataset into training and validation sets (e.g., 80% training, 20% validation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create DataLoader for training and validation
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Hyperparameters
input_dim = 64 * 64 * 3  # For 64x64 RGB images (adjust if needed)
hidden_dim = 256
num_layers = 2
output_dim = len(dataset.classes)  # Number of ASL classes
learning_rate = 0.001
num_epochs = 10
epsilon = 0.2  # Epsilon for epsilon-greedy exploration

# Model initialization
model = PredRNN(input_dim, hidden_dim, num_layers, output_dim, epsilon)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        batch_X = batch_X.view(batch_X.size(0), -1).unsqueeze(1)  # Flatten images and add time dimension

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs[:, -1, :], batch_y)  # Use the last time step for classification
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch_X, batch_y in val_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        batch_X = batch_X.view(batch_X.size(0), -1).unsqueeze(1)  # Flatten images and add time dimension
        outputs = model(batch_X)
        _, predicted = torch.max(outputs[:, -1, :], 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# Save the model
model_save_path = 'model.pt'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Prediction on a new image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Open image and convert to RGB
    image = transform(image)  # Apply transformations
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Load the model
model = PredRNN(input_dim, hidden_dim, num_layers, output_dim, epsilon)
model.load_state_dict(torch.load('model.pt'))  # Load trained model (update path)
model.eval()  # Set the model to evaluation mode

# Example image path
image_path = "asl_dataset\0\hand1_0_bot_seg_1_cropped.jpeg"  # Update this with the correct path

# Preprocess the image and predict
image = preprocess_image(image_path).to(device)
image = image.view(image.size(0), -1).unsqueeze(1)  # Flatten and add time dimension

with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs[:, -1, :], 1)
    predicted_label = dataset.classes[predicted.item()]  # Get the predicted class label

print(f"Predicted ASL letter: {predicted_label}")