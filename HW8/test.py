import model
import torch
import dataset
from torch.utils.data import RandomSampler, SequentialSampler,DataLoader
import numpy as np
import json
import pandas as pd
with open("./config.json",'r',encoding='utf_8') as f:
    congfig_param = json.load(f)

eval_batch_size = congfig_param["test_batchsize"]
model_type = congfig_param["model_type"]

test = np.load("/home_nfs/mingjie.xia/dataset/HW8data/testingset.npy", allow_pickle=True)
data = torch.tensor(test, dtype=torch.float32)
test_dataset = dataset.CustomTensorDataset(data)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=eval_batch_size, num_workers=1)

eval_loss = torch.nn.MSELoss(reduction='none')

checkpoint_path = f'./last_model_{model_type}.pt'
mymodel = torch.load(checkpoint_path,weights_only=False)
mymodel.eval()

# prediction file 
out_file = 'prediction.csv'

anomality = []
with torch.no_grad():
    for i,data in enumerate(test_dataloader):
        img = data.float().cuda()

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