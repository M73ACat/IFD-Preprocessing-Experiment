'''
Author: M73ACat
Date: 2023-04-06 16:19:02
LastEditors: M73ACat
LastEditTime: 2023-04-14 20:15:31
Description: 
Copyright (c) 2023 by M73ACat, All Rights Reserved. 
'''

import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tkinter as tk
from tkinter.filedialog import askopenfilename, askdirectory
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.fftpack import fft, hilbert
from PyEMD import EEMD
from SymplecticGeometryModeDecomposition import SGMD

from keras.layers import Conv1D, Dense, Input, MaxPooling1D, Dropout, Flatten, BatchNormalization, GlobalAveragePooling1D
from keras.models import Model, load_model
from keras.optimizers import Nadam, Adam
from keras.utils import plot_model
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def global_vb():
    global path
    path = os.path.dirname(os.path.abspath(__file__))

def time_record(func):
    def wrapper(*args,**kwards):
        time_st = time.time()
        func(*args,**kwards)
        print(time.time()-time_st)
    return wrapper

def file_select(mode=1):
    """ 1: file 2: dictionary"""
    root = tk.Tk()
    root.withdraw()
    return askopenfilename() if mode == 1 else askdirectory()

def mat2npy():
    path = file_select(2)
    print(path)
    for file in os.listdir(path):
        if 'mat' not in file:
            continue
        print(file)
        mat_date = scio.loadmat(path+'/'+file)
        key = [i for i in mat_date.keys() if 'DE_time' in i]
        mat_signal = mat_date[key[0]]
        print(len(mat_signal))
        mat_signal = np.reshape(mat_signal,(mat_signal.shape[0],))
        np.save(path+'/'+file.replace('mat','npy'),mat_signal)

def dataset_maker_4type(mode):
    path = file_select(2)
    print(path)
    dataset = []
    labels = []
    labels_dic = {
        'Normal':[1,0,0,0],
        'IR':[0,1,0,0],
        'B':[0,0,1,0],
        'OR':[0,0,0,1]
    }
    name_dic = {1:'',2:'_freq'}
    for f in os.listdir(path):
        if 'npy' not in f:
            continue
        sig = np.load('%s/%s'%(path,f))
        print(f,sig.shape)
        label = [k for i,k in labels_dic.items() if i in f][0]
        for i in range(len(sig)//2048):
            temp_sig = sig[i*2048:(i+1)*2048]
            if mode == 1:
                dataset.append(temp_sig)
            else:
                sig_fft = fft(temp_sig)
                dataset.append(abs(sig_fft[:len(sig_fft)//2]))
            labels.append(label)
        print(len(dataset),len(labels))
    dataset = np.array(dataset)
    labels = np.array(labels)
    print(dataset.shape,labels.shape)
    np.savez('%s/%s'%(path,'dataset%s.npz'%name_dic[mode]),dataset=dataset,labels=labels)

def dataset_maker_10type(mode):
    path = file_select(2)
    print(path)
    dataset = []
    labels = []
    labels_dic = {
        'Normal':[1,0,0,0,0,0,0,0,0,0],
        'IR007': [0,1,0,0,0,0,0,0,0,0],
        'IR014': [0,0,1,0,0,0,0,0,0,0],
        'IR021': [0,0,0,1,0,0,0,0,0,0],
        'B007':  [0,0,0,0,1,0,0,0,0,0],
        'B014':  [0,0,0,0,0,1,0,0,0,0],
        'B021':  [0,0,0,0,0,0,1,0,0,0],
        'OR007': [0,0,0,0,0,0,0,1,0,0],
        'OR014': [0,0,0,0,0,0,0,0,1,0],
        'OR021': [0,0,0,0,0,0,0,0,0,1]
    }
    name_dic = {1:'',2:'_freq'}
    for f in os.listdir(path):
        if 'npy' not in f:
            continue
        sig = np.load('%s/%s'%(path,f))
        print(f,sig.shape)
        label = [k for i,k in labels_dic.items() if i in f][0]
        for i in range(len(sig)//2048):
            temp_sig = sig[i*2048:(i+1)*2048]
            if mode == 1:
                dataset.append(temp_sig)
            else:
                sig_fft = fft(temp_sig)
                dataset.append(abs(sig_fft[:len(sig_fft)//2]))
            labels.append(label)
        print(len(dataset),len(labels))
    dataset = np.array(dataset)
    labels = np.array(labels)
    print(dataset.shape,labels.shape)
    np.savez('%s/%s'%(path,'dataset%s.npz'%name_dic[mode]),dataset=dataset,labels=labels)

class models:
    def __init__(self,datasets,inputs,epochs,labels=4,save_path=None) -> None:
        x_train, x_test, y_train, y_test = train_test_split(MinMaxScaler().fit_transform(datasets['dataset']),datasets['labels'],test_size=0.2,shuffle=True,random_state=1024)
        x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.2,shuffle=True,random_state=1024)
        self.train_data = (x_train,y_train)
        self.test_data = (x_test,y_test)
        self.val_data = (x_val,y_val)
        self.inputs = inputs
        self.epochs = epochs
        self.labels = labels
        self.save_path = save_path
    
    def bp_nn(self):
        x_input = Input(shape=(self.inputs,))
        x = Dense(1024,activation='relu')(x_input)
        x = Dropout(0.05)(x)
        x = Dense(512,activation='relu')(x)
        x = Dense(512,activation='relu')(x)
        x = Dropout(0.05)(x)
        x = Dense(256,activation='relu')(x)
        x = Dense(256,activation='relu')(x)
        x = Dropout(0.05)(x)
        x = Dense(64,activation='relu')(x)
        x = Dense(self.labels,activation='softmax')(x)
        self.model = Model(inputs=x_input,outputs=x)
        self.model.compile(optimizer=Nadam(lr=1e-5),loss='categorical_crossentropy',metrics=['accuracy'])
        self.model.summary()
        self.model_name = 'bpnn'

    def cnn1d(self):
        x_input = Input(shape=(self.inputs,1))
        x = Conv1D(32,3,activation='relu',padding='same')(x_input)
        x = Conv1D(32,3,activation='relu',padding='same')(x)
        x = MaxPooling1D(2,strides=2)(x)

        x = BatchNormalization()(x)
        x = Conv1D(64,3,activation='relu',padding='same')(x)
        x = Conv1D(64,3,activation='relu',padding='same')(x)
        x = MaxPooling1D(2,strides=2)(x)

        x = BatchNormalization()(x)
        x = Conv1D(128,3,activation='relu',padding='same')(x)
        x = Conv1D(128,3,activation='relu',padding='same')(x)
        x = MaxPooling1D(2,strides=2)(x)

        x = BatchNormalization()(x)
        x = Conv1D(256,3,activation='relu',padding='same')(x)
        x = Conv1D(256,3,activation='relu',padding='same')(x)
        x = MaxPooling1D(2,strides=2)(x)

        x = Flatten()(x)
        x = Dropout(0.05)(x)
        x = Dense(128,activation='relu')(x)
        x = Dense(self.labels,activation='softmax')(x)
        self.model = Model(inputs=x_input,outputs=x)
        optimizers = Nadam(lr=1e-5)
        self.model.compile(optimizer = optimizers,loss = 'categorical_crossentropy',metrics=['accuracy'])
        self.model.summary()
        self.model_name = 'cnn1d'

    def train(self):
        history = self.model.fit(x=self.train_data[0],y=self.train_data[1],batch_size=20,epochs=self.epochs,validation_data=self.val_data,verbose=2)
        np.savez('%s/%s/history_%s.npz'%(path,self.save_path,self.model_name),loss=history.history['loss'],acc=history.history['accuracy'],val_loss=history.history['val_loss'],val_acc=history.history['val_accuracy'])
        self.model.save('%s/%s/model_%s.h5'%(path,self.save_path,self.model_name))
        with open('%s/%s/%s_%s.txt'%(path,self.save_path,self.model_name,' '.join((str(history.history['loss'][-1]),str(history.history['accuracy'][-1]),str(history.history['val_loss'][-1]),str(history.history['val_accuracy'][-1])))),'a') as r:
            r.write('')
        self.valid()

    def valid(self):
        pred_label = []
        if 'cnn' in self.model_name:
            for data in self.test_data[0]:
                pred_label.append(self.model.predict(np.expand_dims(data,0))[0])
            np.save('%s/%s/1dcnn_pred_label.npy'%(path,self.save_path),pred_label)
        else:
            for data in self.test_data[0]:
                pred_label.append(self.model.predict(np.expand_dims(data,0)))
            np.save('%s/%s/bp_pred_label.npy'%(path,self.save_path),pred_label)
        np.save('%s/%s/true_label.npy'%(path,self.save_path),self.test_data[1])

def cm_maker(pred_label_path=None,labels=None):
    # plt.rc('font',family='Times New Roman',size='14')
    pred_data = np.load(pred_label_path if pred_label_path else file_select(1))
    dir_path, file_name = os.path.split(pred_label_path)
    print('now path is: %s, and the file is: %s'%(dir_path,file_name))
    true_data = np.load(dir_path+'/true_label.npy')
    true_label = [np.argmax(i) for i in true_data]
    pred_label = [np.argmax(i) for i in pred_data]
    cm = confusion_matrix(true_label,pred_label)

    ind_array = np.arange(len(labels))
    cm_copy = np.zeros(cm.shape)
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        cm_copy[y_val][x_val] = cm[y_val][x_val]/sum(cm[y_val])*100
        c = round(cm_copy[y_val][x_val],1)
        plt.text(x_val, y_val, c, color='red', va='center', ha='center')

    print(np.diag(cm_copy))
    print('Means: %s'%np.mean(np.diag(cm_copy)))

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    ax = plt.gca()
    cb = plt.colorbar()

    tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
    cb.locator = tick_locator
    cb.set_ticks([0,np.max(cm)//2//2,np.max(cm)//2,(np.max(cm)+np.max(cm)//2)//2,np.max(cm)])
    cb.set_ticklabels(['0.0%', '25.0%','50.0%', '75.0%', '100.0%'])

    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels)
    plt.yticks(xlocations, labels, rotation=90)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    
    tick_marks = np.array(range(len(labels))) + 0.5
    ax.set_xticks(tick_marks, minor=True)
    ax.set_yticks(tick_marks, minor=True)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.grid(True, which='minor', linestyle='-')
    
    # plt.tight_layout()
    plt.subplots_adjust(top=0.980,bottom=0.09,left=0.0,right=1.0,hspace=0.2,wspace=0.2)
    plt.show() 

def history_plt():
    file_path = file_select(1)
    print(file_path)
    history = np.load(file_path,allow_pickle=True)
    for i in history.keys():
        if i in ['loss', 'val_loss']:
            plt.plot(list(range(len(history[i]))),history[i]/np.max(history[i]),label=i)
        else:
            plt.plot(list(range(len(history[i]))),history[i],label=i)
        plt.xlim(0,len(history[i]))
    plt.legend()
    plt.tight_layout()
    plt.show()
    [print('%s %s'%(i,history[i][-1]),end=' ') for i in history.keys()]
    print('\n')

def signal_fft(_signal):
    _signal = np.abs(fft(_signal))
    length = np.int32(len(_signal))
    _signal = _signal[0:np.int32(length/2)] / length
    _signal[1:-1] *= 2
    return _signal

def envelope(_signal):
    return signal_fft(np.sqrt(hilbert(_signal)**2+_signal**2))

def sig_plot(_sig):
    maxes = np.max(_sig)
    mines = np.min(_sig)
    maxes += maxes/10
    mines += mines/10
    for j in range(_sig.shape[0]):
        plt.subplot(int(_sig.shape[0]+1),1,j+1)
        plt.plot(_sig[j])
        plt.ylabel('(%s)'%(chr(ord('a')+j)))
        plt.xlim(0,len(_sig))
        plt.ylim(mines,maxes)
    plt.show()

def sgmd_show():
    local_path = 'E:\\dataset\\CWRU\\FaultOnly\\'
    sig_dic = {
        'Health': 'Normal_3.npy',
        # 'Health': 'Normal_2.npy',
        # 'Outer Ring': 'OR021@6_3.npy',
        # 'Inner Ring': 'IR021_3.npy',
        # 'Ball': 'B021_3.npy'
        # 'Ball': 'B021_2.npy'
    }
    fault_dic = {
        # 'Health': 1730/60,
        'Health': 1750/60,
        'Outer Ring': 3.5848*1730/60,
        'Inner Ring': 5.4152*1730/60,
        # 'Ball': 4.7135*1730/60
        'Ball': 4.7135*1750/60
    }
    plt.rc('font',family='Times New Roman',size='14')
    for i in sig_dic.keys():
        print(i)
        print(fault_dic[i])
        sig = np.load(local_path+sig_dic[i])

        plt.plot(sig)
        plt.show()

        # seed_number = np.random.randint(1000,1000000,1)[0]
        
        seed_number = 914382
        np.random.seed(seed_number)
        print(seed_number)


        index = np.random.randint(0,len(sig)-8192,1)[0]
        sig = sig[index:index+8192]
        print(len(sig))
        fs = 48000
        sgmd = SGMD(sig,fs,nfft=64,threshold_corr=0.95,threshold_nmse=0.75)
        SGCs = sgmd.sgmd()

        # sig_plot(sgmd.matrix_Y)
  
        print(SGCs.shape)
        maxes = max(max(sig),np.max(SGCs))
        mines = min(min(sig),np.min(SGCs))
        maxes += maxes/10
        mines += mines/10
        plt.figure()
        plt.subplot(int(SGCs.shape[0]+1),1,1)
        plt.plot(sig)
        plt.ylabel('(a)')
        plt.xlim(0,len(sig))
        plt.ylim(mines,maxes)
        for j in range(SGCs.shape[0]):
            plt.subplot(int(SGCs.shape[0]+1),1,j+2)
            plt.plot(SGCs[j])
            plt.ylabel('(%s)'%(chr(ord('a')+j+1)))
            plt.xlim(0,len(sig))
            plt.ylim(mines,maxes)

        plt.figure()
        f = np.array(list(range(int(SGCs.shape[1]/2)))) * fs / SGCs.shape[1]
        plt.subplot(int(SGCs.shape[0]+1),1,1)
        plt.plot(f,envelope(sig))
        # maxes = max(envelope(sig)[1:200])
        # maxes += maxes/10
        # mines = -0.01
        plt.ylabel('(a)')
        # plt.ylim(mines,maxes)
        # plt.xlim(0,250)
        # plt.xlim(0,200)
        for j in range(SGCs.shape[0]):
            plt.subplot(int(SGCs.shape[0]+1),1,j+2)
            plt.plot(f,envelope(SGCs[j]))
            plt.ylabel('(%s)'%(chr(ord('a')+j+1)))
            # plt.xlim(0,250)
            # plt.xlim(0,200)
            # plt.ylim(mines,maxes)

        plt.show()

        from PyEMD import EEMD
        eemd = EEMD(parallel=4)
        eimfs = eemd(sig)

        eimfs = eimfs[:3,:] if eimfs.shape[0] > 3 else eimfs
        plt.figure()
        maxes = max(max(sig),np.max(eimfs))
        mines = min(min(sig),np.min(eimfs))
        maxes += maxes/10
        mines += mines/10
        plt.subplot(int(eimfs.shape[0]+1),1,1)
        plt.plot(sig)
        plt.ylabel('(a)')
        plt.xlim(0,len(sig))
        plt.ylim(mines,maxes)
        for j in range(eimfs.shape[0]):
            plt.subplot(int(eimfs.shape[0]+1),1,j+2)
            plt.plot(eimfs[j])
            plt.ylabel('(%s)'%(chr(ord('a')+j+1)))
            plt.xlim(0,len(sig))
            plt.ylim(mines,maxes)

        plt.figure()
        f = np.array(list(range(int(eimfs.shape[1]/2)))) * fs / eimfs.shape[1]
        plt.subplot(int(eimfs.shape[0]+1),1,1)
        plt.plot(f,envelope(sig))
        maxes = max(envelope(sig)[1:200])
        maxes += maxes/10
        mines = -0.01
        plt.ylabel('(a)')
        # plt.xlim(0,250)
        plt.xlim(0,200)
        plt.ylim(mines,maxes)
        for j in range(eimfs.shape[0]):
            plt.subplot(int(eimfs.shape[0]+1),1,j+2)
            plt.plot(f,envelope(eimfs[j]))
            plt.ylabel('(%s)'%(chr(ord('a')+j+1)))
            # plt.xlim(0,250)
            plt.xlim(0,200)
            plt.ylim(mines,maxes)
        # plt.show()

        plt.figure()
        SGCs = SGCs[0,:]
        eimfs = np.sum(eimfs[:3,:],axis=0)
        maxes = max(np.max(sig),max(np.max(SGCs),np.max(eimfs)))
        mines = min(np.min(sig),min(np.min(SGCs),np.min(eimfs)))
        maxes += maxes/10
        mines += mines/10
        plt.subplot(3,1,1)
        plt.plot(sig)
        plt.ylabel('(a)')
        plt.xlim(0,len(sig))
        plt.ylim(mines,maxes)
        plt.subplot(3,1,2)
        plt.plot(eimfs)
        plt.ylabel('(b)')
        plt.xlim(0,len(sig))
        plt.ylim(mines,maxes)
        plt.subplot(3,1,3)
        plt.plot(SGCs)
        plt.ylabel('(c)')
        plt.xlim(0,len(sig))
        plt.ylim(mines,maxes)

        plt.figure()
        f = np.array(list(range(int(len(sig)/2)))) * fs / len(sig)
        plt.subplot(3,1,1)
        plt.plot(f,envelope(sig))
        maxes = max(envelope(sig)[1:200])
        maxes += maxes/10
        mines = -0.01
        plt.ylabel('(a)')
        plt.xlim(0,200)
        plt.ylim(mines,maxes)
        plt.subplot(3,1,2)
        plt.plot(f,envelope(eimfs))
        plt.ylabel('(b)')
        plt.xlim(0,200)
        plt.ylim(mines,maxes)
        plt.subplot(3,1,3)
        plt.plot(f,envelope(SGCs))
        plt.ylabel('(c)')
        plt.xlim(0,200)
        plt.ylim(mines,maxes)
        plt.show()

def sgmd_test(file_path=None):
    dataset = np.load(file_path if file_path else file_select(1))
    sig = dataset['dataset']
    # index = int(np.random.randint(0,sig.shape[0],1)[0])
    
    index = 3072
    # index = 2466
    
    print(index)
    sig = sig[index]
    fs = 48000
    sgmd = SGMD(sig,fs,nfft=64,threshold_corr=0.95,threshold_nmse=0.95)
    SGCs = sgmd.sgmd()

    # if SGCs.shape[0] > 30:
    #     print(SGCs.shape)
    # else:
    plt.rc('font',family='Times New Roman',size='14')
    plt.figure()
    plt.subplot(int(SGCs.shape[0]+1),1,1)
    plt.plot(sig)
    plt.ylabel('sig')
    plt.xlim(0,len(sig))
    for i in range(SGCs.shape[0]):
        plt.subplot(int(SGCs.shape[0]+1),1,i+2)
        plt.plot(SGCs[i])
        plt.ylabel('sgc%s'%(i+1))
        plt.xlim(0,len(sig))

    plt.figure()
    f = np.array(list(range(int(SGCs.shape[1]/2)))) * fs / SGCs.shape[1]
    plt.subplot(int(SGCs.shape[0]+1),1,1)
    plt.plot(f,envelope(sig))
    plt.ylabel('sig')
    plt.xlim(0,200)
    # plt.xlim(0,1000)
    for i in range(SGCs.shape[0]):
        plt.subplot(int(SGCs.shape[0]+1),1,i+2)
        plt.plot(f,envelope(SGCs[i]))
        plt.ylabel('sgc%s'%(i+1))
        plt.xlim(0,200)
        # plt.xlim(0,1000)

    plt.show()

    # eemd = EEMD()
    # eimfs = eemd(sig)
    # plt.figure()
    # for i in range(eimfs.shape[0]):
    #     plt.subplot(int(eimfs.shape[0]),1,i+1)
    #     plt.plot(eimfs[i])
    #     plt.xlim(0,len(sig))
    # # plt.show()

def sgmd_feature(file_path=None):
    dataset = np.load(file_path if file_path else file_select(1))
    sig = dataset['dataset']
    for i in range(sig.shape[0]):
        sgmd = SGMD(sig[i,:],48000,nfft=64,mode='eig',threshold_corr=0.95,threshold_nmse=0.95)
        sig[i,:] = sgmd.sgmd()[0,:]
    np.savez('%s/sgmd_%s'%(path,os.path.split(file_path)[1]),dataset=sig,labels=dataset['labels'])
    print('done')

def eemd_feature(file_path=None):
    datasets = np.load(file_path if file_path else file_select(1))
    dataset = datasets['dataset']
    eemd_dataset = np.zeros(dataset.shape)
    labels = datasets['labels']
    eemd = EEMD(100)
    for i in range(eemd_dataset.shape[0]):
        eimfs = eemd(dataset[i])
        eemd_dataset[i] = np.sum(eimfs[:3,:],axis=0)
    np.savez('%s/eemd_%s'%(path,os.path.split(file_path)[1]),dataset=eemd_dataset,labels=labels)

if __name__ == '__main__':
    global_vb()

    """ mat转换npy """
    # mat2npy()

    """ 四分类数据集制作 """
    # dataset_maker_4type(1)
    # dataset_maker_4type(2)

    """ 十分类数据集制作 """
    # dataset_maker_10type(1)
    # dataset_maker_10type(2)

    """ 四分类时域数据bp、cnn模型训练 """
    # dataset_path = path+'/dataset_Exp1.npz'
    # model = models(np.load(dataset_path),2048,2,labels=4,save_path='四类时域')
    # model.bp_nn()
    # model.train()
    # model.cnn1d()
    # model.train()

    """ 十分类时域数据bp、cnn模型训练 """
    # dataset_path = path+'/dataset_Exp2.npz'
    # model = models(np.load(dataset_path),2048,100,labels=10,save_path='十类时域')
    # model.bp_nn()
    # model.train()
    # model.cnn1d()
    # model.train()

    """ 四分类频域数据bp、cnn模型训练 """
    # dataset_path = path+'/dataset_Exp1_freq.npz'
    # model = models(np.load(dataset_path),1024,100,labels=4,save_path='四类频域')
    # model.bp_nn()
    # model.train()
    # model.cnn1d()
    # model.train()

    """ 十分类频域数据bp、cnn模型训练 """
    # dataset_path = path+'/dataset_Exp2_freq.npz'
    # model = models(np.load(dataset_path),1024,100,labels=10,save_path='十类频域')
    # model.bp_nn()
    # model.train()
    # model.cnn1d()
    # model.train()

    """ eemd分解和数据制作 """
    # feature_extract('%s/dataset_Exp1.npz'%path)
    # feature_extract('%s/dataset_Exp2.npz'%path)


    """ eemd数据四分类时域数据bp、cnn模型训练 """
    # dataset_path = path+'/eemd_dataset_Exp1.npz'
    # model = models(np.load(dataset_path),2048,100,labels=4,save_path='EEMD四类时域')
    # model.bp_nn()
    # model.train()
    # model.cnn1d()
    # model.train()

    """ eemd十分类时域数据bp、cnn模型训练 """
    # dataset_path = path+'/eemd_dataset_Exp2.npz'
    # model = models(np.load(dataset_path),2048,100,labels=10,save_path='EEMD十类时域')
    # model.bp_nn()
    # model.train()
    # model.cnn1d()
    # model.train()

    """ 训练历史绘图 """
    # history_plt()
    # history_plt()

    """ 混淆矩阵绘制 """
    # cm_maker(path+'/四类时域/1dcnn_pred_label.npy',labels=['Normal','IR','B','OR'])
    # cm_maker(path+'/四类时域/bp_pred_label.npy',labels=['Normal','IR','B','OR'])

    # cm_maker(path+'/四类频域/1dcnn_pred_label.npy',labels=['Normal','IR','B','OR'])
    # cm_maker(path+'/四类频域/bp_pred_label.npy',labels=['Normal','IR','B','OR'])

    # cm_maker(path+'/十类时域/1dcnn_pred_label.npy',labels=['Normal','IR07','IR14','IR21','B07','B14','B21','OR07','OR14','OR21'])
    # cm_maker(path+'/十类时域/bp_pred_label.npy',labels=['Normal','IR07','IR14','IR21','B07','B14','B21','OR07','OR14','OR21'])

    # cm_maker(path+'/十类频域/1dcnn_pred_label.npy',labels=['Normal','IR07','IR14','IR21','B07','B14','B21','OR07','OR14','OR21'])
    # cm_maker(path+'/十类频域/bp_pred_label.npy',labels=['Normal','IR07','IR14','IR21','B07','B14','B21','OR07','OR14','OR21'])

    # cm_maker(path+'/EEMD四类时域/1dcnn_pred_label.npy',labels=['Normal','IR','B','OR'])
    # cm_maker(path+'/EEMD四类时域/bp_pred_label.npy',labels=['Normal','IR','B','OR'])

    # cm_maker(path+'/EEMD十类时域/1dcnn_pred_label.npy',labels=['Normal','IR07','IR14','IR21','B07','B14','B21','OR07','OR14','OR21'])
    # cm_maker(path+'/EEMD十类时域/bp_pred_label.npy',labels=['Normal','IR07','IR14','IR21','B07','B14','B21','OR07','OR14','OR21'])

    """ SGMD分解和数据集制作 """
    # sgmd_feature(path+'/dataset_Exp1.npz')
    # sgmd_feature(path+'/dataset_Exp2.npz')

    """ sgmd四分类时域数据bp、cnn模型训练 """
    # dataset_path = path+'/sgmd_dataset_Exp1.npz'
    # model = models(np.load(dataset_path),2048,100,labels=4,save_path='EEMD四类时域')
    # model.bp_nn()
    # model.train()
    # model.cnn1d()
    # model.train()

    """ sgmd十分类时域数据bp、cnn模型训练 """
    # dataset_path = path+'/sgmd_dataset_Exp2.npz'
    # model = models(np.load(dataset_path),2048,100,labels=10,save_path='EEMD十类时域')
    # model.bp_nn()
    # model.train()
    # model.cnn1d()
    # model.train()

    # history_plt()
    # history_plt()

    """ 混淆矩阵 """
    # cm_maker(path+'/SGMD四类时域/1dcnn_pred_label.npy',labels=['Normal','IR','B','OR'])
    # cm_maker(path+'/SGMD四类时域/bp_pred_label.npy',labels=['Normal','IR','B','OR'])

    # cm_maker(path+'/SGMD十类时域/1dcnn_pred_label.npy',labels=['Normal','IR07','IR14','IR21','B07','B14','B21','OR07','OR14','OR21'])
    # cm_maker(path+'/SGMD十类时域/bp_pred_label.npy',labels=['Normal','IR07','IR14','IR21','B07','B14','B21','OR07','OR14','OR21'])