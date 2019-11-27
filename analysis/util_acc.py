import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os
import json



def save_pic_and_acc(path):
    import os
    if not os.path.exists(path + '/log.txt'):
        print("no log.txt in " + path)
        return 0
    data_temp = []
    with open(path+'/log.txt', 'r') as fdata:
        title = fdata.readline()
        while True:
            line = fdata.readline()
            if not line:
                break
            data_temp.append([float(i) for i in line.split()])
    data = np.array(data_temp)
    try:
        test_acc = data[:, 4]
        train_acc = data[:, 3]
        test_loss = data[:, 2]
        train_loss = data[:, 1]

    except:
        best_acc = 0
        #print(best_acc)
        return 0



    index = np.argmax(test_acc)
    index_train = np.argmax(train_acc)

    if not os.path.exists(path + 'best_acc.txt'):
        filename = os.path.join(path, 'best_acc.txt')
        file_handle = open(filename, mode='w')
        file_handle.write(str(test_acc[index]))

    print(path+': test- '+str(test_acc[index])+'  train- ' + str(train_acc[index_train]))




    import os
    if not os.path.exists(path + '/result.png'):
        #plt.figure(figsize=[14,7])
        plt.subplot(1,2,1)
        plt.plot(train_loss,label = "train loss")
        plt.plot(test_loss,label = "test_loss")
        #plt.legend()
        plt.subplot(1,2,2)
        plt.plot(train_acc,label = "train_acc")
        plt.plot(test_acc,label = "test_acc")
        #plt.legend()

        plt.scatter(index,test_acc[index],edgecolors='r',label="best_acc "+str(test_acc[index]))
        #print("best_acc："+str(test_acc[index]))
        plt.legend()
        #plt.show()
        plt.savefig(path+'/result.png')

    return test_acc[index]




checkfilename = "log.txt"

function = save_pic_and_acc

def eachFile(filepath):
    pathDir = os.listdir(filepath)      #获取当前路径下的文件名，返回List
    pathDir.sort()

    #print(pathDir)
    for s in pathDir:
        newDir=os.path.join(filepath,s)     #将文件命加入到当前文件路径后面
        if os.path.isfile(newDir) :         #如果是文件
            if checkfilename in newDir:     #判断是否是txt
                function(filepath)
        else:
            eachFile(newDir)                #如果不是文件，递归这个文件夹的路径

if __name__ == "__main__":
    import sys
    eachFile(sys.argv[1]
             )