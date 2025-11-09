from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torchvision.transforms as transforms
import torch


class CustomTensorDataset(TensorDataset):
    def __init__(self,tensors):
        self.tensors = tensors
        if tensors.shape[-1] == 3:
            self.tensors = tensors.permute(0,3,1,2) #对数据维度进行交换，N H W 3 -> N 3 W H

        self.transform = transforms.Compose([
            transforms.Lambda(lambda x : x.to(torch.float32)),
            transforms.Lambda(lambda x : 2.* x/255. - 1.)
        ])

    def __len__(self):
        return len(self.tensors)
    
    def __getitem__(self, index):
        x = self.tensors[index]
        if self.transform:
            x = self.transform(x)

        return x
