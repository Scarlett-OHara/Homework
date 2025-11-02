import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
#from torch.utils.tensorboard.writer import SummaryWrite
import numpy as np
import os


batch_size = 64
dataset_dir = "/root/autodl-tmp/food11"
model_save_dir = "/root/HW3/best_param.ckpt"
result_save_dit = "/root/HW3/submission.csv"

test_tfm = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
train_tfm = transforms.Compose([
    transforms.RandomResizedCrop(128,scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    ])

class FoodDataset(Dataset):
    def __init__(self,path,files = None,tfm=test_tfm):
        super().__init__() #调用父类初始化方法，初始化父类里的一些参数
        self.path = path #数据集目录
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")]) #对数据进行排序得到一堆文件名
        if files != None:
            self.files = files
        self.transform = tfm

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index): #Dataloader取图片时会调用此函数
        #return super().__getitem__(index)
        fname = self.files[index]
        img = Image.open(fname)
        img = self.transform(img)
        try:
            label = int(fname.split("/")[-1].split("_")[0]) #得到图片的编号（对应类别）
        except:
            label = -1

        return img,label
    

def Load_DataSet(dataset_dir:str = dataset_dir):
    train_set =    FoodDataset(os.path.join(dataset_dir,"training"),tfm = train_tfm)
    #train_loader = DataLoader(train_set,batch_size = batch_size,shuffle = True,num_workers=0, pin_memory=True)
    valid_set =    FoodDataset(os.path.join(dataset_dir,"validation"), tfm=test_tfm)
    #valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_set =     FoodDataset(os.path.join(dataset_dir,"test"),tfm = test_tfm)
    #test_loader  = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=0, pin_memory=True)
    cross_dataset = ConcatDataset([train_set,valid_set])
    #cross_loader = DataLoader(cross_dataset,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True)
    return cross_dataset,test_set

def Split_train_valid(cross_dataset:DataLoader,k:int):
    dataloader_index = np.arange(len(cross_dataset))
    np.random.shuffle(dataloader_index)
   #print(dataloader_index,len(dataloader_index))
    start = k*2660
    end   = min((k+1)*2660,13296)
    valid_index = dataloader_index[start:end]
    train_index = np.concatenate((dataloader_index[:start],dataloader_index[end:]))
    cross_valid_set = Subset(cross_dataset,valid_index)
    cross_train_set = Subset(cross_dataset,train_index)
    cross_train_loader = DataLoader(cross_train_set,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True)
    cross_valid_loader = DataLoader(cross_valid_set,batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=True)
    return cross_train_loader,cross_valid_loader