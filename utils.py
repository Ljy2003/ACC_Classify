import torch,torchvision
import torch.nn as nn
import numpy as np

class net_base(nn.Module):
    def __init__(self):
        super(net_base,self).__init__()
        self.backbone=None
    
    def freeze(self):
        length=len(list(self.parameters()))
        for i,param in enumerate(self.backbone.parameters()):
            if i < length-20:
                param.requires_grad = False
        
class resnet_18(net_base):
    def __init__(self):
        super(resnet_18,self).__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1000,512),
            nn.ReLU(),
            nn.Linear(512,2)
        )

    def forward(self,x):
        x = self.backbone(x)
        return(self.classifier(x))

class resnet_50(net_base):
    def __init__(self):
        super(resnet_50,self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1000,512),
            nn.ReLU(),
            nn.Linear(512,2)
        )
        
    def forward(self,x):
        x = self.backbone(x)
        return(self.classifier(x))

class resnet_101(net_base):
    def __init__(self):
        super(resnet_101,self).__init__()
        self.backbone = torchvision.models.resnet101(pretrained=True)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1000,512),
            nn.ReLU(),
            nn.Linear(512,2)
        )

    def forward(self,x):
        x = self.backbone(x)
        return(self.classifier(x))
    
model={'resnet_18':resnet_18,'resnet_50':resnet_50,'resnet_101':resnet_101}
