#!/usr/local/anaconda3/bin/python3.6  
# -*- coding:utf-8 -*-

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import sys, os
import numpy as np
import dataload
#import networks
from opts import parse_opts

opt = parse_opts()

model_name = opt.model
epochs = opt.epochs
batch_size = opt.batch_size 
works = opt.num_works
learning_rate = opt.lr
step_rate = opt.step_rate
step_period = opt.step_period
stream = opt.stream
threthold = opt.threthold
IMAGE_PATH = opt.image_path
data_dir = opt.jpg_path
optim = opt.optimizer
result_path = opt.result_path
start_time = opt.start_time
frame_rate=opt.fps
gpu = opt.gpu
## Our video have variety. 森とかビルとか環境が多様なので
## So, we need extract  beforehand, "learning part" and "evaluation part" from each video. 事前に学習用と評価用の画像を決定しておく
annotation_test = opt.annotation_file+"_test.txt"
annotation_train = opt.annotation_file+"_train.txt"

def e_is(lr):
    ## Count 0; in 0.00002 out 2e-5
    lr = float(lr)
    i = 0
    while lr < 1:
        lr *= 10
        i += 1
    return("{}e-{}".format(int(lr), i))
    

if step_rate != 1.0:
    addtext = "_sr-{}_sp-{}".format(int(step_rate*10), step_period)
#corename = opt.save_name+"_"+opt.annotation_file.split("/")[-1]+"_bc-"+str(batch_size)+"_lr-"+str(str(int(learning_rate**(-1))).count("0"))+addtext
corename = model_name+"_"+stream+"_"+optim+"_"+opt.annotation_file.split("/")[-1]+"_"+str(frame_rate)+"fps"+"_bs-"+str(batch_size)+"_lr-"+e_is(learning_rate)+"_"+start_time

transform_train = transforms.Compose(
    [transforms.Resize((256, 256)),
     transforms.RandomHorizontalFlip(p=0.5), 
     transforms.RandomCrop((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # imagenet
     ])
# transform_train = transforms.Compose(
#     [transforms.RandomCrop((224), padding=16),
#      transforms.ToTensor(),
#      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # imagenet
#      ])
transform_test = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # imagenet
    ])


texts = "\nepoch={}, batch_size={}, num_works={}, lr={}, threthold={}, optimizer={}, gpu={}\n"
print("\nLog/"+corename)
print(texts.format(epochs, batch_size, works, learning_rate, threthold, optim, gpu),  "Annotation file: ", opt.annotation_file, "\n")



classes = {'bark':0,'cling':1,'command':2,'eat-drink':3,'look_at_handler':4,'run':5,'see_victim':6,'shake':7,'sniff':8,'stop':9,'walk-trot':10}



## Model
classes_num = 11

if model_name == "resnet":
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, classes_num)
else:
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(4096, classes_num)
print(model)
#exit()

## GPU
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
if gpu >= 0:
    gpu_num = opt.gpu
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_num) #,2,3"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if gpu <= -1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model) # make parallel
    #cudnn.benchmark = True
model = model.to(device)


## Load dataset.
#train_dataset = dataload.MulchVideoset(annotation_train, data_dir, classes, transform_train)
train_dataset = dataload.RGBStream(annotation_train, data_dir, classes, transform_train, fps=frame_rate)
test_dataset  = dataload.RGBStream(annotation_test,  data_dir, classes, transform_test, fps=frame_rate)

#train_size = int(0.8 * len(dataset))
#test_size = len(dataset)-train_size  
#train, test = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=works)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=works)




def display(train_loader, namae):
    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        
        # if title is not None:
        #     plt.title(title)
        # plt.pause(0.001)  # pause a bit so that plots are updated
        plt.imsave(namae+'.jpg', inp)
    # Get a batch of training data
    inputs, labels = next(iter(train_loader))
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    #imshow(out)

# display(train_loader, opt.annotation_file.split("/")[-1])
display(train_loader, corename)

def nofk(output, gt_labels, threthold=0.5):
    return(np.where(output.cpu() < threthold, 0, 1))



def train(train_loader, learning_rate):
    model.train()
    running_loss = 0
    

    

    for batch_idx, (images, labels) in enumerate(train_loader):
        #images = images.transpose(1, 3) # (1,224,224,3) --> (1,3,224,224)
        images = images.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(images)
        
        #loss = criterion(outputs, labels)
        loss = 1*F.multilabel_soft_margin_loss(outputs, labels.cuda(non_blocking=True).float())

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        #
    #####
        
    train_loss = running_loss / len(train_loader)
    return train_loss



def test(test_loader):
    model.eval()
    running_loss = 0
    relevant = [0]*classes_num # Count ground trues for calculate the Recall
    selected = [0]*classes_num # Count selected elements for calculate the Precision
    true_positive = [0]*classes_num  # Count true positive relevant for the Recall and Precision
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            #images = images.transpose(1, 3) # (1,224,224,3) --> (1,3,224,224)
            # images = Variable(images)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            #loss = criterion(outputs, labels)
            loss = 1*F.multilabel_soft_margin_loss(outputs, labels.cuda(non_blocking=True).float())
            running_loss += loss.item()

            labels = labels.cpu().numpy()
            sigmoided = F.sigmoid(outputs)

            predicted = nofk(sigmoided, labels, threthold=threthold)
            #print((labels * predicted).tolist())
            #true_positives.extend((labels * predicted).tolist())
            relevant += np.sum(labels, axis=0)
            selected += np.sum(predicted, axis=0)
            true_positive += np.sum((labels * predicted), axis=0)

    ##
    val_loss = running_loss / len(test_loader)
    
    return val_loss, true_positive, relevant, selected
 

loss_list = []
val_loss_list = []
precision_list = []
recall_list = []
true_positives = []
relevants = []
selecteds = []

oldloss = 2

#criterion = nn.CrossEntropyLoss()
if optim == "sgd":
    print("optimizer = SGD momentum")
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)    
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)




scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_period, gamma=step_rate)

for epoch in range(epochs):
    scheduler.step()
    loss = train(train_loader, learning_rate)
    val_loss, true_positive, relevant, selected = test(test_loader)
    #precision = true_positive/relevant
    #recall = true_positive/selected
    micro_precision = np.sum(true_positive)/np.sum(relevant)
    micro_recall = np.sum(true_positive)/np.sum(selected)

    print('epoch %d, loss: %.4f, val_loss: %.4f, micro_precision: %.4f, micro_recall: %.4f, true_positive: %s, relevant: %s, selected: %s' % (epoch, loss, val_loss, micro_precision, micro_recall, list(true_positive), list(relevant), list(selected)))

    ## logging
    loss_list.append(loss)

    val_loss_list.append(val_loss)
    precision_list.append(micro_precision)
    recall_list.append(micro_recall)
    #true_positives.append(true_positive)
    #relevants.append(relevant)
    #selecteds.append(selected)

    plt.figure()
    fig, ax = plt.subplots(1, 2,figsize=(8,4)) #, sharey=True)
    ax[0].set_ylim([0,1.2])
    ax[1].set_ylim([0,1.2])

    x = []
    for i in range(0, len(loss_list)):
        x.append(i)
    x = np.array(x)
    ax[0].plot(x, np.array(loss_list), label="train")
    ax[0].plot(x, np.array(val_loss_list), label="test")
    #plt.plot(x, np.array(val_acc_list), label="acc")
    ax[0].legend() # 凡例
    plt.xlabel("epoch")
#    plt.ylabel("loss")
    ax[0].set_title("loss")
    ax[1].set_title("micro accuracy")

 #   plt.savefig(os.path.join(IMAGE_PATH,corename+'.png'))

 #   plt.figure()
    ax[1].plot(x, np.array(precision_list), label="precision")
    ax[1].plot(x, np.array(recall_list), label="recall", linestyle="dashed")
    ax[1].legend() # 凡例
    #plt.xlabel("epoch")
    #plt.ylabel("accuracy")
    plt.savefig(os.path.join(IMAGE_PATH,corename+'_accuracy.png'))

    ### Save a model.
    if val_loss < oldloss:
        torch.save(model.state_dict(), os.path.join(result_path,corename+'.ckpt'))
        print("save to "+os.path.join(result_path,corename+'.ckpt'))
        oldloss = val_loss

    
print('Finished Training')
plt.figure()
x = []
for i in range(0, len(loss_list)):
    x.append(i)

print("save to "+corename+".png")

### Save a model.
#torch.save(model.state_dict(), os.path.join(result_path+corename+'.ckpt'))
#print("save to "+os.path.join(result_path+corename+'.ckpt'))


exit()

classes = ['bark','cling','comand','eat','handlr','run','victim','shake','sniff','stop','walk']

