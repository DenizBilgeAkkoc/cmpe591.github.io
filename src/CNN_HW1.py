import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#to load data and split
img_tensor = (torch.tensor(np.load("before-image-1000.npy").astype(np.float32), dtype=torch.float32))/255.0
act_tensor = torch.tensor(np.load("actions-1000.npy"), dtype=torch.float32)
loc_tensor= torch.tensor(np.load("after-object-location-1000.npy"), dtype=torch.float32)

def rgb_to_grayscale_luminance(image):
    # Apply the luminance formula: Y = 0.299*R + 0.587*G + 0.114*B
    grayscale_image = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
    return grayscale_image.unsqueeze(1)  # Add a channel dimension for grayscale

img = rgb_to_grayscale_luminance(img_tensor)

img_train=img[0:800]
img_test=img[800:1000]

act_train=act_tensor[0:800]
act_test=act_tensor[800:1000]

loc_train=loc_tensor[0:800]
loc_test=loc_tensor[800:1000]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        self.conv3=nn.Conv2d(16, 8, 3, padding=1)

        # Calculate the size after pooling 128->64>32>16
        pooling_output_size = 16 * 16 * 8  #16x16 and 8 channels
        self.fc1 = nn.Linear(pooling_output_size, 120)
        self.fc2 = nn.Linear(120+4, 128)#+4 means adding action data
        self.fc3 = nn.Linear(128, 64)
        self.fc4=nn.Linear(64,2)

    def forward(self, x, y):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=0)
        x = F.relu(self.fc1(x))

        if y == 0:#adding action data
            extra_tensor = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=x.dtype)
        elif y == 1:
            extra_tensor = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=x.dtype)
        elif y == 2:
            extra_tensor = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=x.dtype)
        else:
            extra_tensor = torch.tensor([1.0, 0.0,0.0, 0.0], dtype=x.dtype)

        # Concatenate the extra tensor to the flattened tensor x
        x = torch.cat((x, extra_tensor), dim=0)

        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x=self.fc4(x)
        return x
    
net = Net()

criterion = nn.L1Loss()#used L1 in training but used both to analyz at the end
criterion2=nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(300):  # loop over the dataset 300 times
    running_loss = 0.0
    for i in range(800):
        # get the inputs and labels
        inputs1=img_train[i]
        inputs2=act_train[i] 
        labels =loc_train[i]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs1,inputs2)
        loss = criterion(outputs, labels)#this is L1
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()


    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 800:.3f}')
    running_loss = 0.0

print('Finished Training')

totalValLoss=0
totalValLoss2=0
with torch.no_grad():
    # set the model in evaluation mode
    net.eval()
    # loop over the validation set
    for j in range(200):

        # make the predictions and calculate the validation loss
        pred = net(img_test[j],act_test[j])
        a=criterion2(pred, loc_test[j])
        a2=criterion(pred, loc_test[j])
        totalValLoss += a
        totalValLoss2+=a2

print("L1")
print(totalValLoss2/200)

print("MSE")
print(totalValLoss/200)

torch.save({
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'final_l1_loss': totalValLoss2/200,
    'final_mse_loss': totalValLoss/200,
}, 'final_model.pth')


