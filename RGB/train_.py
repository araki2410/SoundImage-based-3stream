#classes = ['bark','cling','comand','eat','handlr','run','victim','shake','sniff','stop','walk']

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
import architecture
#from opts import parse_opts

def e_is(lr):
    ## Count 0; in 0.00002 out 2e-5
    lr = float(lr)
    i = 0
    while lr < 1:
        lr *= 10
        i += 1
    return("{}e-{}".format(int(lr), i))




class Train:
    def __init__(self, opt):
        model_name = opt.model
        self.epochs = opt.epochs
        batch_size = opt.batch_size 
        works = opt.num_works
        self.learning_rate = opt.lr
        step_rate = opt.step_rate
        step_period = opt.step_period
        stream = opt.stream
        self.threthold = opt.threthold
        self.IMAGE_PATH = opt.image_path
        data_dir = opt.jpg_path
        optim = opt.optimizer
        self.result_path = opt.result_path
        start_time = opt.start_time
        frame_rate=opt.fps
        gpu = opt.gpu
        ## Our video have variety. 森とかビルとか環境が多様なので
        ## So, we need extract  beforehand, "learning part" and "evaluation part" from each video. 事前に学習用と評価用の画像を決定しておく
        annotation_test = opt.annotation_file+"_test.txt"
        annotation_train = opt.annotation_file+"_train.txt"
        
    ### Show Sample images        
        # display(train_loader, corename)




        if step_rate != 1.0:
            addtext = "_sr-{}_sp-{}".format(int(step_rate*10), step_period)
        #corename = opt.save_name+"_"+opt.annotation_file.split("/")[-1]+"_bc-"+str(batch_size)+"_lr-"+str(str(int(learning_rate**(-1))).count("0"))+addtext
        self.corename = model_name+"_"+stream+"_"+optim+"_"+opt.annotation_file.split("/")[-1]+"_"+str(frame_rate)+"fps"+"_bs-"+str(batch_size)+"_lr-"+e_is(self.learning_rate)+"_"+start_time
        self.trained_image_name = os.path.join(self.IMAGE_PATH, self.corename+'.png')
        self.trained_model_name = os.path.join(self.result_path, self.corename+'.ckpt')
    
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


#        texts = "\nepoch={}, batch_size={}, num_works={}, lr={}, threthold={}, optimizer={}, gpu={}\n"
#        print("\nLog/"+self.corename)
#        print(texts.format(epochs, batch_size, works, learning_rate, threthold, optim, gpu),  "Annotation file: ", opt.annotation_file, "\n")



        classes = {'bark':0,'cling':1,'command':2,'eat-drink':3,'look_at_handler':4,'run':5,'see_victim':6,'shake':7,'sniff':8,'stop':9,'walk-trot':10}

## Model
        self.classes_num = 11

        if model_name == "resnet":
            self.model = models.resnet18(pretrained=True)
            self.model.fc = nn.Linear(512, self.classes_num)
        elif model_name == "myrgb":
            self.model = architecture.RGB_stream()
        else:
            self.model = models.vgg16(pretrained=True)
            self.model.classifier[6] = nn.Linear(4096, self.classes_num)
        print(self.model)
#exit()

## GPU
        #device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if gpu >= 0:
            gpu_num = opt.gpu
            os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_num) #,2,3"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if gpu <= -1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model) # make parallel
            #cudnn.benchmark = True
        self.model = self.model.to(self.device)


## Load dataset.
        #train_dataset = dataload.MulchVideoset(annotation_train, data_dir, classes, transform_train)
        train_dataset = dataload.RGBStream(annotation_train, data_dir, classes, transform_train, fps=frame_rate)
        test_dataset  = dataload.RGBStream(annotation_test,  data_dir, classes, transform_test, fps=frame_rate)

        #train_size = int(0.8 * len(dataset))
        #test_size = len(dataset)-train_size  
        #train, test = torch.utils.data.random_split(dataset, [train_size, test_size])
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=works)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=works)

        # DISPLAY()???

        if optim == "sgd":
            print("optimizer = SGD momentum")
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)    
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_period, gamma=step_rate)




    def display(train_loader, namae):
        ### Show Sample images        
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
        inputs, labels, filenames = next(iter(train_loader))
        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)
        #imshow(out)


    # def nofk(self, output, gt_labels, threthold=0.5):
    def nofk(self, output, threthold=0.5):
        ### Convert tensor([[0.1, 0.5, ... 0.4, 0.4]]
        ### To => [[0 1 ... 0 0]]
        return(np.where(output.cpu() < threthold, 0, 1))



    def train(self, train_loader, learning_rate):
        self.model.train()
        running_loss = 0
    
        for batch_idx, (images, labels, filenames) in enumerate(train_loader):
            #images = images.transpose(1, 3) # (1,224,224,3) --> (1,3,224,224)
            images = images.to(self.device)
            labels = labels.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward + backward + optimize
            outputs = self.model(images)
        
            #loss = criterion(outputs, labels)
            loss = 1*F.multilabel_soft_margin_loss(outputs, labels.cuda(non_blocking=True).float())
        
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            #
        #####
        
        train_loss = running_loss / len(train_loader)
        return train_loss



    def test(self, test_loader):
        self.model.eval()
        running_loss = 0
        total_GT = [0]*self.classes_num # Count ground trues for calculate the Recall
        total_selected = [0]*self.classes_num # Count selected elements for calculate the Precision
        total_TP = [0]*self.classes_num  # Count true positive relevant for the Recall and Precision
        predict_data = []
        with torch.no_grad():
            for batch_idx, (images, labels, filenames) in enumerate(test_loader):
                #images = images.transpose(1, 3) # (1,224,224,3) --> (1,3,224,224)
                # images = Variable(images)
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)

                #loss = criterion(outputs, labels)
                loss = 1*F.multilabel_soft_margin_loss(outputs, labels.cuda(non_blocking=True).float())
                running_loss += loss.item()

                labels = labels.cpu().numpy()
                sigmoided = F.sigmoid(outputs)

                # predicted_label = self.nofk(sigmoided, labels, threthold=self.threthold)
                predicted_label = self.nofk(sigmoided, threthold=self.threthold)
                
                #true_positives.extend((labels * predicted_label).tolist())
                total_GT += np.sum(labels, axis=0)
                total_selected += np.sum(predicted_label, axis=0)
                total_TP += np.sum((labels * predicted_label), axis=0)
                predict_data.append(zip(filenames, labels, predicted_label))
        ##
        val_loss = running_loss / len(test_loader)
    
        ###### val_loss, total-numof-TP,total-numof-GT,TP+FP #,([(image_name, image_GT, image_TP+FP)],..)
        return val_loss, total_TP, total_GT, total_selected, predict_data
 



    def run(self):

        loss_list = []
        val_loss_list = []
        precision_list = []
        recall_list = []
        true_positives = []
        relevants = []
        selecteds = []

        oldloss = 2

        
        for epoch in range(self.epochs):
            self.scheduler.step()
            loss = self.train(self.train_loader, self.learning_rate)
            val_loss, true_positive, relevant, selected, predicted_data = self.test(self.test_loader)
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
            # plt.ylabel("loss")
            ax[0].set_title("loss")
            ax[1].set_title("micro accuracy")

            # plt.savefig(os.path.join(IMAGE_PATH,corename+'.png'))

            # plt.figure()
            ax[1].plot(x, np.array(precision_list), label="precision")
            ax[1].plot(x, np.array(recall_list), label="recall", linestyle="dashed")
            ax[1].legend() # 凡例
            #plt.xlabel("epoch")
            #plt.ylabel("accuracy")
            plt.savefig(self.trained_image_name)

        ### Save a model.
            if val_loss < oldloss:
                torch.save(self.model.state_dict(), self.trained_model_name)
                print("save to "+ self.trained_model_name)
                oldloss = val_loss

        print('Finished Training')

        plt.figure()
        x = []
        for i in range(0, len(loss_list)):
            x.append(i)

        print("save to "+self.corename+".png")
        ### Save a model.
        #torch.save(model.state_dict(), os.path.join(result_path+corename+'.ckpt'))
        #print("save to "+os.path.join(result_path+corename+'.ckpt'))



