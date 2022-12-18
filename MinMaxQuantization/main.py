import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

# Load ViT-B/16 model
model = torchvision.models.vit_b_16(pretrained=True)

# MinMax Scaling Quantization using QuantStub
model = torch.quantization.QuantStub(model)
model.eval()

# Load Dataset
train_dataset = torchvision.datasets.ImageNet(root='data/', train=True, download=True)
test_dataset = torchvision.datasets.ImageNet(root='data/', train=False, download=True)

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Loss fn and Optimizer
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train
for epoch in range(10):
    for inp, labels in train_dataloader:
        optim.zero_grad()
        outputs = model(inp)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optim.step()

    correct = 0
    total = 0
    with torch.no_grad():
        for inp, labels in test_dataloader:
            outputs = model(inp)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    
    print(f'Epoch {epoch+1}: Test accuracy = {correct / total:.4f}')
