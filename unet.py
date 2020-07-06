import os # get all the files inside a folder
from PIL import Image # read images
import numpy as np # arrays are great
import time # measure how long training takes

# these 3 are necessary to design any neural network
import torch
import torch.nn as nn
import torch.nn.functional as F

# load data
from torch.utils.data import Dataset, DataLoader

# convenient transforms to match the expected format
import torchvision
from torchvision.transforms import ToTensor, Resize

# gradient descent
import torch.optim as optim


# let's define the way our data (images & masks) have
# to be read and transformed to tensors
class MyDataset(Dataset):
    def __init__(self, root):
        
        img_path = os.path.join(root, 'images')
        mask_path = os.path.join(root, 'masks')
        
        files = [file for file in os.listdir(img_path) if not file.startswith('.')]
        # just make sure corresponding masks have same name
        
        imgs = []
        masks = []
        for file in files:
            imgs.append(Image.open(os.path.join(img_path, file)).convert('RGB'))
            masks.append(Image.open(os.path.join(mask_path, file)))
            
        self.imgs = imgs
        self.masks = masks
            
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        mask = self.masks[idx]
        img = ToTensor()(img)
        mask = torch.as_tensor(np.array(mask)[None], dtype=torch.float32)
        return img, mask

root = 'data/train/'
trainset = MyDataset(root)

# it's a good thing to check if the data has the right shape and data type

#first_img, first_mask = trainset[0]

#print(first_img.shape) # should be torch.Size([C, H, W])
#print(first_img.dtype) # should be torch.float32

#print(first_mask.shape) # should be torch.Size([nb_classes, H, W])
#print(first_mask.dtype) # should be torch.float32




# U-Net architecture

# I took this part from https://github.com/milesial/Pytorch-UNet/tree/master/unet

# --- --- ---
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
# --- --- ---

# it's a good thing to check if data can pass through the network

#batch = 7 # arbitrary
#print(UNet(n_channels=3, n_classes=1)(torch.rand((batch,3,256,256))).shape)
# should be torch.Size([batch, 1, 256, 256])

model = UNet(n_channels=3, n_classes=1) # create instance

# get data
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, drop_last=True)

# if there's a GPU available then it's going to be used (on a server for example)
# otherwise let's use the CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device) # move it

# training mode
model.train()

# loss
criterion = nn.MSELoss()

# optim
optimizer = optim.SGD(model.parameters(), lr=0.1)




# train model

t0 = time.time()
loss_track = []
for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(trainloader):

        inputs, targets = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    loss_track.append(running_loss)
    print('Epoch ',epoch+1,' Loss ',running_loss)

elapsed = time.time() - t0
print("Time : {} sec ({}) min".format(int(np.round(elapsed)),np.round(elapsed/60,1)))

# evaluation time

model.eval() # eval mode

## metric

def IoU(y, y_hat):
    '''Intersection over Union
       good metric for segmentation
    '''
    inter = y & y_hat
    union = y | y_hat
    return inter.sum() / union.sum()

## eval on train set

trainloader = DataLoader(trainset)

train_IoU = []
for data in trainloader:
    inputs, targets = data[0].to(device), data[1].to(device)
    with torch.no_grad():
        outputs = model(inputs)
    outputs = outputs[0,0,...].cpu().numpy().round().astype(bool)
    targets = targets[0].cpu().numpy().astype(bool)
    train_IoU.append(IoU(outputs, targets))
print(f"IoU train = {np.round(np.array(train_IoU).mean()*100)}%")

## eval on validation set

root = 'data/val'
valset = MyDataset(root)
valloader = DataLoader(valset)

val_IoU = []
for data in valloader:
    inputs, targets = data[0].to(device), data[1].to(device)
    with torch.no_grad():
        outputs = model(inputs)
    outputs = outputs[0,0,...].cpu().numpy().round().astype(bool)
    targets = targets[0].cpu().numpy().astype(bool)
    val_IoU.append(IoU(outputs, targets))
print(f"IoU val = {np.round(np.array(val_IoU).mean()*100)}%")

# test

img_test_np = np.array(Resize(256)(Image.open('data/test/1.png')))
img_test = ToTensor()(img_test_np)[None]
with torch.no_grad():
    outputs = model(img_test.to(device))
mask = outputs[0,0,...].cpu().numpy().round().astype(bool)
pos = np.where(mask==False)
img_test_np[pos[0], pos[1], :] = 0

import matplotlib.pyplot as plt
save_path = 'unet_seg_result.png'
plt.imsave(save_path, img_test_np)

print('Test image has been saved.')