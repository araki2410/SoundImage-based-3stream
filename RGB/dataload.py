import torch
import torchvision.transforms as transforms
import os, re
from PIL import Image
import numpy as np
import librosa


class RGBStream(torch.utils.data.Dataset):
    def __init__(self, annotation_file, data_dir, classes, transform=None, fps=6):
        ##
        ## For load 1/n frame
        ##
        n = int(30/6)
        self.image_dataframe = []
        self.transform = transform
        
        f = open(annotation_file)
        ## classes = {"bark":0,  "cling":1, "command":2,
        ##            "eat-drink":3, "look_at_handler":4,
        ##            "run":5, "see_victim":6, "shake":7,
        ##            "sniff":8, "stop":9, "walk-trot":10}
        i = 0
        for aline in f:
            i += 1
            if i % n == 0:
                match = re.search(r'\d+ \d+_\d+.jpg .*', aline)
                video = match.group(0).split(" ")[0] # 2017mmdd
                frame = match.group(0).split(" ")[1] # 2017mmdd_0000.jpg
                ml_class =  match.group(0).split(" ")[2:] # walk-trot command cling ...
                label = [0]*len(classes) #[0,0,0,0,0,0,0,0,0,0,0]
                for aclass in ml_class:
                    try:
                        label[classes[aclass]] = 1 #[0,1,1,0,0,0,0,0,0,0,1]
                    except:
                        pass
                ##
                #print(os.path.join(data_dir, video, frame), label)
                self.image_dataframe.append([os.path.join(data_dir, video, frame), label])

            
    def __len__(self):
        return len(self.image_dataframe)

    def __getitem__(self, idx):
        #dataframeから画像へのパスとラベルを読み出す
        filename = self.image_dataframe[idx][0]
        label = self.image_dataframe[idx][1]
        img_path = os.path.join(filename)
        # #画像の読み込み
        #img_path = "data/UECFOOD100/1/11292.jpg"
        #label = 1
        image = Image.open(img_path)
        # #画像へ処理を加える
        #t = transforms.Resize((32, 32))
        #image = t(image)
        if self.transform:
            image = self.transform(image)

        return image, np.array(label)




