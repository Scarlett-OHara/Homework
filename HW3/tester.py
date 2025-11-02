import torch
import tqdm
import torchvision.models as models
import numpy as np
import dataset
import pandas as pd
import os
import torchvision.transforms as transforms
device = "cuda" if torch.cuda.is_available() else "cpu"
model_save_dir = "/root/HW3/KF-SUB/"
result_save_dir = "/root/HW3/submission_fF.csv"
batch_size = 64
dataset_dir = "/root/autodl-tmp/food11"

myseed = 6666  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

test_tfm = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

test_tfm0 = transforms.Compose([
    transforms.RandomResizedCrop(128,scale=(0.8,1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

test_tfm1 = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

test_tfm2 = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

test_tfm3 = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])


test_set_aug0 = dataset.FoodDataset(os.path.join(dataset_dir,"test"),tfm = test_tfm0)
test_set_aug1 = dataset.FoodDataset(os.path.join(dataset_dir,"test"),tfm = test_tfm1)
test_set_aug2 = dataset.FoodDataset(os.path.join(dataset_dir,"test"),tfm = test_tfm2)
test_set_aug3 = dataset.FoodDataset(os.path.join(dataset_dir,"test"),tfm = test_tfm3)
test_set_nom = dataset.FoodDataset(os.path.join(dataset_dir,"test"),tfm = test_tfm)

test_loader_aug0  = torch.utils.data.DataLoader(test_set_aug0,batch_size=64,shuffle=False,num_workers=0, pin_memory=True)
test_loader_aug1  = torch.utils.data.DataLoader(test_set_aug1,batch_size=64,shuffle=False,num_workers=0, pin_memory=True)
test_loader_aug2  = torch.utils.data.DataLoader(test_set_aug2,batch_size=64,shuffle=False,num_workers=0, pin_memory=True)
test_loader_aug3  = torch.utils.data.DataLoader(test_set_aug3,batch_size=64,shuffle=False,num_workers=0, pin_memory=True)
test_loader_nom  = torch.utils.data.DataLoader (test_set_nom, batch_size=64,shuffle=False,num_workers=0, pin_memory=True)

test_loader_aug = [test_loader_aug0,test_loader_aug1,test_loader_aug2,test_loader_aug3]
def get_model(k:int):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features,11)
    model.load_state_dict(torch.load(model_save_dir+f"parm_fold{k}.ckpt",weights_only=True))
    model = model.to(device)
    model.eval()
    return model

def prediction(model:list,test_loader_nom,test_loader_aug):
    pred_all = []
    with torch.no_grad():
        test_pbar = tqdm.tqdm(test_loader_nom,position=0,leave=True)
        for batch,*batch_aug in zip(test_pbar,*test_loader_aug):
            imgs,_ = batch
            imgs_aug = [img[0] for img in batch_aug]

            test_pred = 0
            test_pred_aug = [0 for i in range(len(imgs_aug))]

            for i in range(len(model)):
                test_pred += model[i](imgs.to(device))

                for j in range(len(imgs_aug)):
                    test_pred_aug[j] += model[i](imgs_aug[j].to(device))

            test_pred  = test_pred / len(model)

            for i in range(len(test_pred_aug)):
                test_pred_aug[i] = test_pred_aug[i] / len(model)

            test_pred_aug = sum(test_pred_aug) / len(test_pred_aug)

            test_pred_all = 0*test_pred + 1*test_pred_aug

            test_label = np.argmax(test_pred_all.detach().cpu().numpy(),axis=1)
            pred_all += test_label.squeeze().tolist()

            test_pbar.set_description(f"Testing Progress")        

    return pred_all
print("------start testing------")
model0 = get_model(0)

model1 = get_model(1)
model2 = get_model(2)
model3 = get_model(3)
model4 = get_model(4)

model =  [model0,model1,model2,model3,model4]
#model =  [model0]
pred_all =  prediction(model,test_loader_nom,test_loader_aug)

print("------end testing------")

print("------saving csv------")
def pad4(i):
    return "0"*(4-len(str(i))) + str(i)
df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(1,len(test_set_nom)+1)]
df["Category"] = pred_all
df.to_csv(result_save_dir,index = False)
print("------end------")