import copy
import numpy as np
import os
import numpy
import torch
import random
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.init as nninit
from torch.nn.parameter import Parameter
import torchvision.transforms.functional as TF

import time
from PIL import Image
import torchattacks2

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

def linf_distance(images, adv_images):
    diff = abs(images-adv_images.cpu())
    maxlist = []
    for i in diff:
        maxlist.append(torch.max(i))
    return 255*((sum(maxlist)/len(maxlist)).item())

def l2_distance(corrects, images, adv_images, device="cuda"):
    delta = (adv_images - images.to(device)).view(len(images), -1)
    l2 = torch.norm(delta[~corrects], p=2, dim=1).mean()
    return l2.item()

if __name__=="__main__":
    # switch out with adv trained model
    model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
    model.cuda()
    
    model.eval()
    
    # Reimplemented with modified torchattacks library
    atk = torchattacks2.PGD(model, eps=8/255., alpha=2/255., steps=7)
    #atk = torchattacks2.PGD(model, eps=8/255., alpha=2/255., steps=20)
    #atk = torchattacks2.PGD(model, eps=8/255., alpha=2/255., steps=30)
    
    print("-"*100)
    print(atk)
    
    correct = 0
    total = 0
    avg_act = 0
    counter = 0
    
    start = time.time()
    
    for images, labels in data_loader:
        labels = labels.to(device)
        images = images.to(device)
        adv_images = images
        
        # Uncomment to attack loader images
        adv_images = atk(images, labels)
        
        total += len(labels)
        
        with torch.no_grad():
            out = model(adv_images)

        act,pred = out.max(1, keepdim=True)
        corrects = (pred.view_as(labels) == labels)
        correct += corrects.sum().cpu()
        avg_act += act.sum().data

        l2 = l2_distance(corrects, images, adv_images, device=device)
        linf = linf_distance(images, adv_images)
        print("Images: " + str(total) + " | Robust Acc: " + str(100* float(correct)/float(total)) + "% | L2 Distance: " + str(round(l2, 3)) + " | Linf Distance: " + str(round(linf)))
        
        if total >= 100:
            break

    print('Total elapsed time (sec) : %.2f' % (time.time() - start))
    print('Robust accuracy: %.2f %%' % (100 * float(correct) / total))
    
    acc = 100. * float(correct) / len(data_loader.dataset),100. * float(avg_act) / len(data_loader.dataset)
    
    print('Final Accuracy: ', acc)

