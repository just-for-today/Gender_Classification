import torchvision.models as models
import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device('cuda:0')#选择GPU
    print("GPU is open and connection")
    
    net.to(device)

else:
    print('no GPU')
    
#加载各个模型
alexnet=torch.load('../input/alexnet-35/gender_alexnet_35.pth')
vgg_16=torch.load('../input/vgg16-25/gender_vgg16_25.pth')
resnet_18=torch.load('../input/resnet-resnet-18-30/gender_resnet_18_30.pth')
resnet_50=torch.load('../input/resnet50-25/gender_resnet_50_25.pth')
resnet_152=torch.load('../input/rennet152-20/gender_resnet_152.pth')

import torch
pred_alexnet_35=[]
pred_resnet_152=[]
pred_resnet_50=[]
pred_resnet_18=[]
pred_vgg_16=[]

with torch.no_grad():
    for data in testloader:
        imageid, image = data['imageid'],data['image'] 
        image=image.to(device)
        
#         alexnet_35
        outputs_alexnet_35 = alexnet_35(image)
        outputs_alexnet_35=outputs_alexnet_35.to(device)
        
        _, predicted_alexnet_35 = torch.max(outputs_alexnet_35, 1)  #返回每行最大值与最大值的索引
        pred_alexnet_35.extend(predicted_alexnet_35.tolist())
        
#         resnet152
        outputs_resnet_152 = resnet_152(image)
        outputs_resnet_152=outputs_resnet_152.to(device)
        
        _, predicted_resnet_152 = torch.max(outputs_resnet_152, 1)  #返回每行最大值与最大值的索引
        pred_resnet_152.extend(predicted_resnet_152.tolist())
        
#         resnet50
        outputs_resnet_50 = resnet_50(image)
        outputs_resnet_50=outputs_resnet_50.to(device)
        
        _, predicted_resnet_50 = torch.max(outputs_resnet_50, 1)  #返回每行最大值与最大值的索引
        pred_resnet_50.extend(predicted_resnet_50.tolist())
        
#         resnet18
        outputs_resnet_18 = resnet_18(image)
        outputs_resnet_18=outputs_resnet_18.to(device)
        
        _, predicted_resnet_18 = torch.max(outputs_resnet_18, 1)  #返回每行最大值与最大值的索引
        pred_resnet_18.extend(predicted_resnet_18.tolist()) 
        
#         vgg16
        outputs_vgg_16 = vgg_16(image)
        outputs_vgg_16=outputs_vgg_16.to(device)
        
        _, predicted_vgg_16 = torch.max(outputs_vgg_16, 1)  #返回每行最大值与最大值的索引
        pred_vgg_16.extend(predicted_vgg_16.tolist())
        
import numpy as np
pred_alexnet_35=np.array(pred_alexnet_35)
pred_resnet_152=np.array(pred_resnet_152)
pred_resnet_50=np.array(pred_resnet_50)
pred_resnet_18=np.array(pred_resnet_18)
pred_vgg_16=np.array(pred_vgg_16)

#向量合并
temp1=np.vstack((pred_alexnet_35,pred_resnet_152))
temp2=np.vstack((temp1,pred_resnet_50))
temp3=np.vstack((temp2,pred_resnet_18))
temp4=np.vstack((temp3,pred_vgg_16))
print(temp4)

#投票
from collections import Counter
pred=np.zeros(temp4.shape[0]) #pred 5,    temp4 5*5708
# print(pred.shape)
end_pred=[]
for i in range(5708):
    for j in range(5):
        pred[j]=temp4[j][i]             
        
        votes=Counter(pred)
#         print(votes)
        
    end_pred.append(votes.most_common(1)[0][0])

end_pred=np.array(end_pred,dtype='int64')
print(type(end_pred[0]))
print(end_pred[:15])

#结果写入文件
test_csv = pd.read_csv('/kaggle/input/gender/test.csv')
submit = pd.DataFrame({'Id':test_csv.id,'label':end_pred})
submit.to_csv("submission2.csv",index=False)
print(submit)
