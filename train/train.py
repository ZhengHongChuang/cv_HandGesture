'''
Descripttion: ResNet50
version: 1.0
Author: zhc
Date: 2023-10-28 14:55:58
LastEditors: zhc
LastEditTime: 2023-10-28 20:59:14
'''
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import json
from torchvision import transforms 
from PIL import Image
import torchvision.models as models


class MyDataset(Dataset):
    def __init__(self,root,train=True,transform=None):
        imgs = []
        with open('config\path2label.json', 'r') as json_file:
            label_mapping = json.load(json_file)
        for path in os.listdir(root):
            path_prefix = path[:3]
            label = label_mapping[path_prefix]
            class_path = os.path.join(root,path)
            for img in os.listdir(class_path):
                img_path = os.path.join(class_path,img)
                imgs.append((img_path,label))
        train_data_list, val_data_list = self.split_dataset(imgs)   
        if train:
            self.imgs = train_data_list
        else:
            self.imgs = val_data_list
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __getitem__(self,index):
        img_path= self.imgs[index][0]
        imgLabel = self.imgs[index][1]
        data = Image.open(img_path)
        if data.mode != 'RGB':
            data = data.convert('RGB')
        data = self.transform(data)
        return data,imgLabel
    def __len__(self):
        return len(self.imgs)
    def split_dataset(self,images):
        val_data_list = images[::5]
        train_data_list = []
        for item in images:
            if item not in val_data_list:
                train_data_list.append(item)
        return train_data_list,val_data_list

if __name__ == '__main__':
    dataset_dir = "datasets"
    batch_size = 64
    epochs = 500
    train_data = MyDataset(dataset_dir,train=True)
    val_data = MyDataset(dataset_dir,train=False)
    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_data,batch_size=batch_size,shuffle=True)

    resnet = models.resnet50(pretrained=True) 
    for parm in resnet.parameters():
        parm.requires_grad = False
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 14),
    )
    criterion = nn.CrossEntropyLoss()   
    optimizer = torch.optim.Adam(resnet.parameters(), lr=0.0001)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    resnet.to(device)
    for epoch in range(epochs):
        print('\nEpoch: %d' % (epoch + 1))

        resnet.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = resnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
        print('Waiting for validation...')
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = resnet(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('Correct: %d' % (correct))
            print('Total: %d' % (total))
            print('Accuracy: %f %%' % (100 * float(correct) / total))
        # 每100个epoch保存一次模型
        if (epoch+1) % 100 == 0:
            torch.save(resnet.state_dict(), 'model/model_%d.pth' % (epoch + 1))
    
