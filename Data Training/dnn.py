import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras import Input,layers
from keras import Sequential, Model
from keras import optimizers
from keras.utils import to_categorical
from tensorflow.python.util import compat
from keras import backend as K
import tensorflow as tf
import os
import sklearn.metrics


# get confusion matrix
def plot_confusion_matrix(cm, labels_name, title,figname):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')
    plt.savefig(figname, format='png')
    plt.show()


def unpickle(file):
    with open(file, 'rb') as fo:
        dct = pickle.load(fo, encoding='bytes')
    return dct


if __name__ == "__main__":
    
    pickle_file='mfcc5'
    category_required =5

    dct = unpickle(pickle_file)
    # trainig data and trainig data labels
    trainig_data = dct["mfcc"]
    trainig_data_labels =dct["labels"]
    print("Different Classes is :",len(set(trainig_data_labels)))
    print(trainig_data.shape)
    print(trainig_data_labels.shape)


    trainig_data_labels = to_categorical(trainig_data_labels,num_classes=category_required)
    trainig_data.astype("float32")
    trainig_data_labels.astype("float32")

    
    trains_nums = int(trainig_data.shape[0]*0.8)
    val_nums = int(trainig_data.shape[0] * 0.2)
    trains_d = trainig_data[0:trains_nums]
    trains_l = trainig_data_labels[0:trains_nums]
    print(trains_d.shape)
    print(trains_l.shape)

    val_d = trainig_data[trains_nums:val_nums+trains_nums]
    val_l = trainig_data_labels[trains_nums:val_nums+trains_nums]


    # graph 
    input_tensor = Input(shape=(13,))
    x = layers.Dense(64,activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(input_tensor)
    x = layers.Dense(128,activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(64,activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(x)
    output_tensor = layers.Dense(category_required,activation='softmax')(x)

    model = Model(input_tensor,output_tensor)
    model.summary()


    model.compile(optimizer='adam',loss='categorical_crossentropy',
                      metrics=['accuracy'])
    
    callback =[
        keras.callbacks.TensorBoard(
            log_dir="log_dir",
            histogram_freq=1,
            embeddings_freq=1,    
        )
    ]
    
    history = model.fit(trains_d,trains_l,epochs=3000,batch_size=1024,validation_data=(val_d,val_l),
    callbacks=callback)

    score = model.evaluate(val_d,val_l)
    print(score)

    model.save("model/")

    val_predict = model.predict(val_d)
    predict_array = np.zeros(shape=(val_predict.shape[0],))
    for index,vec in  enumerate(val_predict):
        outcome = np.argmax(vec)
        predict_array[index] = outcome
    
    ground_true_array = np.zeros(shape=(val_l.shape[0],))
    for index,vec in enumerate(val_l):
        outcome = np.argmax(vec)
        ground_true_array[index] = outcome
    
    print(predict_array.shape)
    print(ground_true_array.shape)
    acc = sklearn.metrics.accuracy_score(ground_true_array,predict_array)
    if category_required==5:
        labels_name =["aa","E","ih","oh","ou"]
        confusion_matrix = sklearn.metrics.confusion_matrix(ground_true_array,predict_array)
        plot_confusion_matrix(cm = confusion_matrix,labels_name=labels_name,title="DNN5,acc is {}"
     .format(round(acc,2)), figname="DNN5")

    elif category_required==9:
        labels_name =["CH","SS","nn","RR","aa","E","ih","oh","ou"]
        confusion_matrix = sklearn.metrics.confusion_matrix(ground_true_array,predict_array)
        plot_confusion_matrix(cm = confusion_matrix,labels_name=labels_name,title="DNN9,acc is {}"
     .format(round(acc,2)), figname="DNN9")
