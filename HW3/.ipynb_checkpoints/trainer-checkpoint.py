# Import necessary packages.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

#设置随机数种子保证结果可复现
class Classifier(nn.Module):
    def __init__(self,):
        super().__init__()
        #input [3,128 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3,64,3,1,1), #in_channel,output_channel,kernel_size,stride,pooling[64,128,128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),    #kernel_size,stride,padding[64,64,64]

            nn.Conv2d(64,128,3,1,1),#[128,64,64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),#[128,32,32]

            nn.Conv2d(128,256,3,1,1),#[256,32,32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),#[256,16,16]

            nn.Conv2d(256,512,3,1,1),#[512,16,16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),#[512,8,8]

            nn.Conv2d(512,512,3,1,1),#[512,8,8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),#[512,4,4]

         )
        
        self.fc = nn.Sequential(
            nn.Linear(512*4*4,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,11)
        )
        #这里没有softmax的原因是,pytorch crossentropy函数中内置了softmax
    
    def forward(self,x):
        out = self.cnn(x) #out中算上了batch [batch,c,w,h]
        out = out.view(out.size()[0], -1) #展平操作后输入全连接层，3维至二维[batch_size,c*w*h]
        return  self.fc(out)

device = "cuda" if torch.cuda.is_available() else "cpu"
patience = 300


def trainer(train_loader:DataLoader,valid_loader:DataLoader,
            model,optimizer,criterion,n_epochs:int,k:int,result:list):
    
    writer = SummaryWriter(log_dir = f"runs/exp_fold{k}")
    stale,best_acc=0,0
    for epoch in range(n_epochs):
        model.train()
        train_loss = []
        train_accs = []
        # 创建训练进度条，只初始化一次
        train_pbar = tqdm(train_loader, position=0, leave=True, desc=f"Train Pro|Fold:{k}| Epoch {epoch+1}/{n_epochs}")
    
        for batch in train_pbar:
            imgs, labels = batch
            logits = model(imgs.to(device))  # Forward pass
            loss = criterion(logits, labels.to(device))
            optimizer.zero_grad()  # 清除上一步梯度
            loss.backward()  # 计算当前梯度
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()  # 更新模型参数

            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            train_loss.append(loss.item())  # 保存标量值
            train_accs.append(acc.item())  # 保存标量值
            # 更新进度条的附加信息
            train_pbar.set_postfix({'train loss': loss.item(), 'train acc': acc.item()})           
        # 计算训练损失和准确度均值
        train_loss_mean = sum(train_loss) / len(train_loss)
        train_accs_mean = sum(train_accs) / len(train_accs)
        writer.add_scalar("Accuracy/Train",train_accs_mean,epoch)

        valid_loss = []
        valid_accs = []
        model.eval()  # 设置模型为评估模式
        valid_pbar = tqdm(valid_loader, position=0, leave=True, desc=f"Valid Pro|Fold{k}| Epoch {epoch+1}/{n_epochs}")

        for batch in valid_pbar:
            imgs, labels = batch
            with torch.no_grad():
                logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            valid_loss.append(loss.item())  # 保存标量值
            valid_accs.append(acc.item())  # 保存标量值
            valid_pbar.set_postfix({'valid loss': loss.item(), 'valid acc': acc.item()})
        
        # 计算验证损失和准确度均值
        valid_loss_mean = sum(valid_loss) / len(valid_loss)
        valid_accs_mean = sum(valid_accs) / len(valid_accs)
        writer.add_scalar("Accuracy/Valid",valid_accs_mean,epoch)
        print(f'[Epoch:{epoch+1:03d}/{n_epochs:03d}] train:loss={train_loss_mean:.5f},acc={train_accs_mean:.5f}|valid:loss={valid_loss_mean:.5f},acc={valid_accs_mean:.5f}')

        # 保存最佳模型
        if valid_accs_mean > best_acc:
            best_acc = valid_accs_mean
            result.append(best_acc)
            print(f'Best model found at epoch:{epoch+1}|Fold:{k}, saving model')
            torch.save(model.state_dict(), f"/root/HW3/result/parm_fold{k}.ckpt")
            stale = 0
        else:
            stale += 1
            if stale > patience:
                print("Training convergence")
                break


