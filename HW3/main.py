import torch
import dataset
import trainer
import torchvision.models as models
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
import torch.utils.tensorboard.summary 
import numpy as np

myseed = 6666  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)


batch_size = 64
dataset_dir = "/root/autodl-tmp/food11"
model_save_dir = "/root/HW3/best_param.ckpt"
result_save_dit = "/root/HW3/submission.csv"
device = "cuda" if torch.cuda.is_available() else "cpu"
n_epochs = 100
patience = 300
k_fold = 5

cross_dataset,test_dataset = dataset.Load_DataSet(dataset_dir)

result_acc=[]
for k in range(k_fold):
    print(f"Fold{k} start")
    train_loader,valid_loader = dataset.Split_train_valid(cross_dataset,k)

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features,11)
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=3e-4,weight_decay=1e-5)

    trainer.trainer(train_loader,valid_loader,model=model,optimizer=optimizer,criterion=criterion,n_epochs=100,k=k,result=result_acc)
    print("acc",result_acc)
    print(f"Fold{k} end")
