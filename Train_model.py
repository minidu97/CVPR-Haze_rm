from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import model as ResINet
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define dataset and dataloader
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(root='./Dataset', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Initialize the model, loss function, and optimizer
model = ResINet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Save the trained model
torch.save(model.state_dict(), 'resinet_trained_model.pth')