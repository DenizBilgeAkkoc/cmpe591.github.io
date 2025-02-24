import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

# Load the data from the file

img_tensor = (torch.tensor(np.load("before-image-1000.npy").astype(np.float32), dtype=torch.float32))/255.0
act_tensor = torch.tensor(np.load("actions-1000.npy"), dtype=torch.float32)
loc_tensor= torch.tensor(np.load("after-object-location-1000.npy"), dtype=torch.float32)

def rgb_to_grayscale_luminance(image):
    # Apply the luminance formula: Y = 0.299*R + 0.587*G + 0.114*B
    grayscale_image = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
    return grayscale_image.unsqueeze(1)  # Add a channel dimension for grayscale

img = rgb_to_grayscale_luminance(img_tensor)
downscaled_images = F.interpolate(img, size=(64,64), mode='bilinear', align_corners=False)#128x128 was too big so made it 64x64

img_train=downscaled_images[0:800].view(800, -1)#flattening the image
img_test=downscaled_images[800:1000].view(200, -1)

act_train=act_tensor[0:800]
act_test=act_tensor[800:1000]

loc_train=loc_tensor[0:800]
loc_test=loc_tensor[800:1000]


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(64*64, 512)
        self.fc2 = nn.Linear(512, 124)
        self.fc3=nn.Linear(124+4, 64)#add action here
        self.fc4=nn.Linear(64, 64)
        self.fc5=nn.Linear(64, 2)


    def forward(self, input1, y):
        input1 = F.relu(self.fc1(input1))
        input1 = F.relu(self.fc2(input1))

        if y == 0:
            extra_tensor = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=input1.dtype)
        elif y == 1:
            extra_tensor = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=input1.dtype)
        elif y == 2:
            extra_tensor = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=input1.dtype)
        else:
            extra_tensor = torch.tensor([1.0, 0.0,0.0, 0.0], dtype=input1.dtype)

        # Concatenate the extra tensor to the flattened tensor x
        input1 = torch.cat((input1, extra_tensor), dim=0)

        input1 = F.relu(self.fc3(input1))
        input1 = F.relu(self.fc4(input1))
        input1 = self.fc5(input1)
        return input1
    


net=MLP()

criterion=nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(800):
        # get the inputs; data is a list of [inputs, labels]
        inputs1=img_train[i]
        inputs2=act_train[i] 
        labels =loc_train[i]
    
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs1,inputs2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 800:.3f}')
    running_loss = 0.0

print('Finished Training')

totalValLoss=0
with torch.no_grad():
    # set the model in evaluation mode
    net.eval()
    # loop over the validation set
    for j in range(200):

        # make the predictions and calculate the validation loss
        pred = net(img_test[j],act_test[j])
        a=criterion(pred, loc_test[j])
        totalValLoss += a


print("msg")
print(totalValLoss/200)

torch.save({
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'final_mse_loss': totalValLoss/200,
}, 'final_mlp_model.pth')


