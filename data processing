from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import transforms, datasets
from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

#data processing
class GenderDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, image_dir, transformss,flag):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transformss = transformss
        self.flag=flag

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        if self.flag=='train':
            if torch.is_tensor(idx):
                idx = idx.tolist()

            img_name = os.path.join(self.image_dir,
                                str(self.csv_file.iloc[idx, 0])+'.jpg')
            image = Image.open(img_name)
#             image = io.imread(img_name)
            label = self.csv_file.iloc[idx, 1]
            label = np.array([label])
            label = label.astype('float')
#             label = label.astype('float').reshape(-1, 2)
            sample = {'image': image, 'landmarks': label}

            image=sample['image']
            label=sample['landmarks']
        
#             print(type(image))
            image =self.transformss(image)

            return {'image': image, 'landmarks': label}
       
        if self.flag=='test':
            if torch.is_tensor(idx):
                idx = idx.tolist()
  
            img_name = os.path.join(self.image_dir,
                                str(self.csv_file.iloc[idx, 0])+'.jpg')
            image = Image.open(img_name) #提取图片
            imageid=self.csv_file.iloc[idx, 0]
#           image = io.imread(img_name)
#           label = self.csv_file.iloc[idx, 1]
#           label = np.array([label])
#           label = label.astype('float')
#           label = label.astype('float').reshape(-1, 2)
            sample = {'imageid': imageid, 'image': image}
            image=sample['image']  
            image =self.transformss(image)

            return {'imageid': imageid, 'image': image}
            
train_img_dir='/kaggle/input/gender/train/train'
train_csv_file='/kaggle/input/gender/train.csv'
test_img_dir='/kaggle/input/gender/test/test'
test_csv_file='/kaggle/input/gender/test.csv'

train_data_transform = transforms.Compose([
        transforms.RandomSizedCrop(200),  #随机裁剪
        transforms.RandomApply([transforms.RandomHorizontalFlip(0.5)],p=0.8),    #随机水平翻转
        transforms.RandomApply([transforms.RandomRotation(2.8)],p=0.8), #一定概率旋转旋转
        transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
test_data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
 
train_full_dataset=GenderDataset(train_csv_file,train_img_dir,train_data_transform,'train')
test_dataset=GenderDataset(test_csv_file,test_img_dir,test_data_transform,'test')

train_size = int(0.9 * len(train_full_dataset))
validation_size = len(train_full_dataset) - train_size
train_dataset, validation_dataset = torch.utils.data.random_split(train_full_dataset, [train_size, validation_size])

print(len(train_full_dataset))
print(len(train_dataset))
print(len(validation_dataset))

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=8,num_workers=4,shuffle=True)  #shuffle=True删除
validationloader = torch.utils.data.DataLoader(validation_dataset, batch_size=8,num_workers=4,shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4,
                                          shuffle=False, num_workers=4)
#图片展示
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize  反规范化
    npimg = img.numpy()
#     print(npimg.shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0))) #改变矩阵索引值
#     print(npimg.shape)
    plt.show()


# get some random training images
dataiter = iter(trainloader)  #迭代
# print(dataiter.next())
data= dataiter.next()


images=data['image']
labels=data['landmarks']
# print(dataiter.next())
print(images.shape)

# show images
imshow(torchvision.utils.make_grid(images))  #将若干张图片拼接成一张图片
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

            
