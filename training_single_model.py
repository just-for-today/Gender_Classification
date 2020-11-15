#手写神经网络
# import torch.nn as nn
# import torch.nn.functional as F


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)  #卷积输入通道 输出通道  卷积核5*5 步长为1 padding=0
#         self.pool = nn.MaxPool2d(2, 2)    #最大池化 2*2,步长为2
#         self.conv2 = nn.Conv2d(6, 16, 5)  #输入通道 输出通道 卷积核5*5
#         self.fc1 = nn.Linear(47*47*16, 120)  #输入特征数  输出特征数
#         self.fc2 = nn.Linear(120, 84)  #输入特征数  输出特征数
#         self.fc3 = nn.Linear(84,2)   #输入特征数  输出特征数
# #         self.fc3 = nn.Linear(84, 10)


#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))  #卷积，激活，池化
#         x = self.pool(F.relu(self.conv2(x)))   #卷积 激活 池化
#         x = x.view(-1,47*47*16)  
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# net = Net()

#迁移学习，训练网络(fine tuning)
import torchvision.models as models
import torch.nn as nn
#resnet18
# resnet=models.resnet18(pretrained=True)
# resnet.fc = nn.Sequential(
#     nn.Dropout(0.2),
#     nn.Linear(512, 2)
# )

#resnet152
# resnet = models.resnet152(pretrained=True)

# resnet.fc=nn.Sequential(
#      nn.Dropout(0.2),
#      nn.Linear(2048, 2)
#  )
# print(resnet)

# resnet=models.resnet50(pretrained=True)
# resnet.fc=nn.Linear(2048,2)
# print(resnet)

# alexnet=models.alexnet(pretrained=True)
# alexnet.classifier[6]= nn.Linear(4096,2)

vgg16=models.vgg16(pretrained=True)
vgg16.classifier[6] = nn.Linear(4096,2)
net=vgg16
print(net)

import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

criterion = nn.CrossEntropyLoss()  #交叉熵
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  #SGD梯度下降
# scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.1, patience=5, verbose=True)
# optimizer = optim.SGD(net.parameters(), lr=0.0001)  #SGD梯度下降  二次训练，
# optimizer = optim.Adam(net.parameters(), lr=0.0001)

if torch.cuda.is_available():
    device = torch.device('cuda:0')#选择GPU
    print("GPU is open and connection")
    
    net.to(device)

else:
    print('no GPU')
    
# 训练模型并在验证集上查看准确率
accuracy_train=[]
accuracy_validation=[]
Loss=[]
Epoch=50
for epoch in range(Epoch):  # loop over the dataset multiple times
    correct_train = 0
    total_train = 0
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data['image'],data['landmarks']
        labels=labels.reshape(8,)
        labels=labels.long()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
#         net=nn.Dropout(0.2)
        inputs=inputs.to(device)        
        outputs = net(inputs)
        
        outputs=outputs.to(device)
        labels=labels.to(device)
        
#         print(outputs.shape)

        loss = criterion(outputs, labels)
        loss=loss.to(device)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        # print statistics
        running_loss += loss.item()
        
#         scheduler.step(running_loss/100)
#         print(get_lr(optimizer))
        
#         print(net.optimizer.state_dict()['param_groups'][0]['lr'])
#         print("lr:",net.scheduler.get_lr())
        
        if (i+1) % 100 == 0:    # print every mini-batches
            print('[%d, %5d] loss: %.3f ' %
                  (epoch + 1, i + 1, running_loss / 100))  #每批次的损失值
            Loss.append(running_loss/100)
            running_loss = 0.0    
    
    #训练集的准确率    
    Accuracy_train=correct_train/total_train    
    print('Accuracy of the network train images: %d %%' % (100 * Accuracy_train))
    accuracy_train.append(Accuracy_train)
    
    #验证集的准确率
    correct_validation = 0
    total_validation = 0
    
    with torch.no_grad():
        for data in validationloader:
            labels, image = data['landmarks'],data['image'] 
            labels=labels.reshape(8,)
            labels=labels.long()
            image=image.to(device)
        
            outputs = net(image)
            outputs=outputs.to(device)
            labels=labels.to(device)
        
            _, predicted = torch.max(outputs, 1)
            total_validation += labels.size(0)
            correct_validation += (predicted == labels).sum().item()
    
    Accuracy_validation=correct_validation / total_validation
    print('Accuracy of the network on the validation images: %d %%' % (100 * Accuracy_validation))
    accuracy_validation.append(Accuracy_validation)
            

print('Finished Training')

#绘制损失值图像
import matplotlib.pyplot as plt
X=[]
for i in range(len(Loss)):
    X.append(i+1)
plt.plot(X,Loss,linewidth=1)
plt.title('train Loss of batch', fontsize=24,color='r') #标题
plt.xlabel('numbers of batch', fontsize=14)  #横坐标标签1
plt.ylabel('accuracy', fontsize=14) #纵坐标标签0 
plt.show()

#绘制训练，验证折线图
import matplotlib.pyplot as plt
X=[]
for i in range(Epoch):
    X.append(i+1)
# print(X)
# print(accuracy)
plt.plot(X, accuracy_train, linewidth=3,color='b',label='accuracy_train')
plt.plot(X,accuracy_validation,linewidth=3,color='g',label='accuracy_validation')

plt.title('train accuracy and validation accuracy of epoch', fontsize=18,color='r') #标题
plt.xlabel('epoch', fontsize=14)  #横坐标标签
plt.ylabel('train and validation accuracy', fontsize=14) #纵坐标标签

plt.legend()
# plt.tick_params(axis='both', labelsize=14) #刻度大小
plt.show()

PATH = './gender_vgg16.pth'
torch.save(net, PATH)

单个vgg16预测
pred=[]
for data in testloader:
    imageid, image = data['imageid'],data['image'] 
    image=image.to(device)
    outputs_vgg_16 = net(image)
    outputs_vgg_16=outputs_vgg_16.to(device)
        
    _, predicted_vgg_16 = torch.max(outputs_vgg_16, 1)  #返回每行最大值与最大值的索引
    pred.extend(predicted_vgg_16.tolist())
print(len(pred))

#提交结果
test_csv = pd.read_csv('/kaggle/input/gender/test.csv')
submit = pd.DataFrame({'Id':test_csv.id,'label':pred})
submit.to_csv("submission2.csv",index=False)
print(submit)
