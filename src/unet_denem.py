from unet_model import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Load and preprocess data
img_tensor = (torch.tensor(np.load("before-image-1000.npy").astype(np.float32), dtype=torch.float32))/255.0
act_tensor = torch.tensor(np.load("actions-1000.npy"), dtype=torch.float32)
loc_tensor = torch.tensor(np.load("after-object-location-1000.npy"), dtype=torch.float32)
after_img_tensor = (torch.tensor(np.load("after-image-1000.npy").astype(np.float32), dtype=torch.float32))/255.0

def rgb_to_grayscale_luminance(image):
    # Apply the luminance formula: Y = 0.299*R + 0.587*G + 0.114*B
    grayscale_image = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
    return grayscale_image.unsqueeze(1)  # Add a channel dimension for grayscale

# Convert to grayscale and split data
img = rgb_to_grayscale_luminance(img_tensor)
after_img = rgb_to_grayscale_luminance(after_img_tensor)
img_train = img[0:800]
img_test = img[800:1000]
act_train = act_tensor[0:800]
act_test = act_tensor[800:1000]
loc_train = loc_tensor[0:800]
loc_test = loc_tensor[800:1000]
after_img_train = after_img[0:800]
after_img_test = after_img[800:1000]

# Initialize network and training components
net = UNet(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training settings
num_epochs = 300
accumulation_steps = 8
running_loss = 0.0

for epoch in range(num_epochs):
    running_loss = 0.0
    optimizer.zero_grad()  # Zero gradients at start of epoch
    
    for i in range(800):
        # Get the inputs and labels
        inputs1 = img_train[i].unsqueeze(0)  # Add batch dimension [1, C, H, W]
        inputs2 = act_train[i]
        inputs2_onehot = F.one_hot(inputs2.to(torch.int64), num_classes=4).float()
        real = after_img_train[i].unsqueeze(0)  # Add batch dimension
        
        # Forward pass
        outputs = net(inputs1, inputs2_onehot)
        loss = criterion(outputs, real) / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Accumulate running loss
        running_loss += loss.item() * accumulation_steps
        
        # Update weights after accumulation_steps samples
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / accumulation_steps:.3f}')
            running_loss = 0.0

print('Finished Training')

# Final validation
totalValLoss = 0

with torch.no_grad():
    net.eval()  # Set the model in evaluation mode
    for j in range(200):
        inputs1 = img_test[j].unsqueeze(0)  # Add batch dimension
        inputs2 = act_test[j]
        inputs2_onehot = F.one_hot(inputs2.to(torch.int64), num_classes=4).float()
        
        pred = net(inputs1, inputs2_onehot)
        totalValLoss += criterion(pred, after_img_test[j].unsqueeze(0))


print("MSE")
print(totalValLoss/200)

# Save model
torch.save({
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'final_mse_loss': totalValLoss/200,
}, 'final_model_unet.pth')