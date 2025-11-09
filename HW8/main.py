import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import RandomSampler, SequentialSampler,DataLoader
from qqdm import qqdm, format_str
import pandas as pd
import model
import dataset
import json

with open("./config.json",'r',encoding='utf_8') as f:
    congfig_param = json.load(f)


def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(48763)


train = np.load("/home_nfs/mingjie.xia/dataset/HW8data/trainingset.npy", allow_pickle=True)

###config
batch_size = congfig_param["train_batchsize"]
num_epochs = congfig_param["num_epochs"]
learning_rate = congfig_param["learning_rate"]
model_type = congfig_param["model_type"]   # selecting a model type from {'cnn', 'fcn', 'vae', 'resnet'}

#load data
x = torch.from_numpy(train) #内存共享机制
train_dataset = dataset.CustomTensorDataset(x)
train_sampler = RandomSampler(train_dataset) #等价于shuffle = True
train_loader  = DataLoader(train_dataset,sampler=train_sampler,batch_size=batch_size)

###define model
model_classes = {'fcn': model.fcn_autoencoder(), 'cnn': model.conv_autoencoder(), 'vae': model.VAE()}
mymodel = model_classes[model_type].cuda()

#define loss
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(mymodel.parameters(), lr=learning_rate)
best_loss = np.inf

mymodel.train()
qqdm_train = qqdm(range(num_epochs), desc=format_str('bold', 'Description'))
for epoch in qqdm_train:
    total_loss = []
    #qqdm_train = qqdm(train_loader, desc=format_str('bold', 'Description'))
    step = 0
    for data in train_loader:
        img = data.float().cuda()
        if model_type == "fcn":
            img = img.view(img.shape[0],-1)
        
        output = mymodel(img)

        if model_type in ['vae']:
            loss = model.loss_vae(output[0], img, output[1], output[2], criterion)
        else:
            loss = criterion(output, img)

        total_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        qqdm_train.set_infos({'epoch': f'{epoch + 1:.0f}/{num_epochs:.0f}','loss': f'{loss:.4f}','batch':f'{step}/{len(train_loader)}'})

    mean_loss = np.mean(total_loss)
    
    if mean_loss < best_loss:
        best_loss = mean_loss
        torch.save(mymodel, './best_model_{}.pt'.format(model_type))

    #qqdm_train.set_infos({'epoch': f'{epoch + 1:.0f}/{num_epochs:.0f}','loss': f'{mean_loss:.4f}',})
    torch.save(mymodel, './last_model_{}.pt'.format(model_type))
'''
eval_batch_size = 200
data = torch.tensor(test, dtype=torch.float32)
test_dataset = dataset.CustomTensorDataset(data)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=eval_batch_size, num_workers=1)
eval_loss = torch.nn.MSELoss(reduction='none')

checkpoint_path = f'./last_model_{model_type}.pt'
mymodel_test = torch.load(checkpoint_path,weights_only=False)
mymodel_test.eval()

# prediction file 
out_file = 'prediction.csv'

anomality = []
with torch.no_grad():
    for i,data in enumerate(test_dataloader):
        data = data.float().cuda()

        if model_type in ['fcn']:
            img = img.view(img.shape[0], -1)

        output = mymodel(img)
        if model_type in ['vae']:
            output = output[0]

        if model_type in ['fcn']:
            loss = eval_loss(output, img).sum(-1)
        else:
            loss = eval_loss(output, img).sum([1, 2, 3])

        anomality.append(loss)

anomality = torch.cat(anomality, axis=0)
anomality = torch.sqrt(anomality).reshape(len(test), 1).cpu().numpy()
df = pd.DataFrame(anomality, columns=['score'])
df.to_csv(out_file, index_label = 'ID')
'''