import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import train_
from train_ import e_is
#from train_ import *
from opts import parse_opts
import time
from itertools import chain

options = parse_opts()

corename = (options.model+"_"+
            options.stream+"_"+
            options.optimizer+"_"+
            options.annotation_file.split("/")[-1]+"_"+
            str(options.fps)+"fps"+
            "_bs-"+str(options.batch_size)+
            "_lr-"+e_is(options.lr)+"_"+
            options.start_time)


predict_data_filename = "./Log/predict_" + corename + ".log"
f = open(predict_data_filename, 'w')

print("\nLog/"+corename)
texts = "\nepoch={}, batch_size={}, num_works={}, lr={}, threthold={}, optimizer={}, gpu={}\n"
print(texts.format(options.epochs, options.batch_size, options.num_works, options.lr, options.threthold, options.optimizer, options.gpu),  "Annotation file: ", options.annotation_file, "\n")

trainer = train_.Train(options)


def run():
    loss_list = []
    val_loss_list = []
    precision_list = []
    recall_list = []
    #true_positives = []
    #relevants = []
    #selecteds = []
    #predict_datas = []

    oldloss = 2
        
    for epoch in range(trainer.epochs):
        trainer.scheduler.step()
        loss = trainer.train(trainer.train_loader, trainer.learning_rate)
        val_loss, true_positive, grand_truth, selected, predict_data = trainer.test(trainer.test_loader)
        #precision = true_positive/relevant
        #recall = true_positive/selected
        micro_precision = np.sum(true_positive)/np.sum(grand_truth)
        micro_recall = np.sum(true_positive)/np.sum(selected)

        print('\rEPOCH %d, loss: %.4f, val_loss: %.4f, micro_precision: %.4f, micro_recall: %.4f                                                                                     '
              % (epoch, loss, val_loss, micro_precision, micro_recall))
        print('true_positive: %s' % (list(true_positive)))
        print('grand_truth: %s' % (list(grand_truth)))
        print('selected: %s' % (list(selected)))
#        print("\u001B[3A", end="") ## Over Write a line

    ### Over Write the terminal
        # print('EPOCH %d, LOSS: %.4f, Val_LOSS: %.4f, Micro_Precision: %.4f, Micro_Recall: %.4f, TP: %s, GT: %s, TP+FP: %s' % (epoch, loss, val_loss, micro_precision, micro_recall, list(true_positive), list(grand_truth), list(selected)))
        # print("\u001B[2A", end="")

    ## logging
        loss_list.append(loss)
    

        val_loss_list.append(val_loss)
        precision_list.append(micro_precision)
        recall_list.append(micro_recall)
        #true_positives.append(true_positive)
        #relevants.append(relevant)
        #selecteds.append(selected)
        for i in predict_data:
            # predict_datas += list(i)
            # (list(map(lambda y : list(map(lambda z : list(z), y)), list(map(lambda x : list(x) , i)))))
            for line in i:
                #print(line[0], list(map(lambda x: list(x), line[1:])))
                f.write(str(line[0]) + str(list(map(lambda x: list(x), line[1:])))+"\n")
            #f.write(txt + "\n")

    ### Plot
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
        ax[0].legend() # 凡例
        plt.xlabel("epoch")
        ax[0].set_title("loss")
        ax[1].set_title("micro accuracy")

        # plt.savefig(os.path.join(IMAGE_PATH,corename+'.png'))

        # plt.figure()
        ax[1].plot(x, np.array(precision_list), label="precision")
        ax[1].plot(x, np.array(recall_list), label="recall", linestyle="dashed")
        ax[1].legend() # 凡例
        #plt.xlabel("epoch")
        #plt.ylabel("accuracy")
        plt.savefig(trainer.trained_image_name)
        plt.close()

    ### Save a model.


        if val_loss < oldloss:
            torch.save(trainer.model.state_dict(), trainer.trained_model_name)
            #print("## Save to "+ trainer.trained_model_name + "                                                                             ")
            #print('true_positive: %s, grand_truth: %s, selected: %s' % (list(true_positive), list(grand_truth), list(selected)))
            print("## Save to "+ trainer.trained_model_name)
            oldloss = val_loss

    return loss_list #, predict_datas




start_time = time.time()
####trainer.run()
loss_list = run()
#loss_list, predict_datas = run()
finish_time = time.time()


#for i in predict_datas:
#    print(list(i))


print('\n\n## Finished Training')

#plt.figure()
#x = []
#for i in range(0, len(loss_list)):
#    x.append(i)

print("## Save to "+trainer.corename+".png")
print("## Save to "+predict_data_filename)
### Save a model.
#torch.save(model.state_dict(), os.path.join(result_path+corename+'.ckpt'))
#print("save to "+os.path.join(result_path+corename+'.ckpt'))

#######


print("## Finish time: ", finish_time - start_time)
