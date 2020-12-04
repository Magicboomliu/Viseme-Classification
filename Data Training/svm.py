import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import dct
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import librosa
import pickle
# SVM libraies
from sklearn import svm
from sklearn import preprocessing
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
    data_dict = unpickle(pickle_file)
    frame_mfcc_data = data_dict["mfcc"]
    frame_label = data_dict["labels"]
    "Try doing SVM in this so called Big Dataset"
    # 划分一波training_set 和 Testing Set
    trains_nums = int(frame_mfcc_data.shape[0]*0.8)
    val_nums = int(frame_mfcc_data.shape[0] * 0.2)
    trains_d = frame_mfcc_data[0:trains_nums]
    trains_l = frame_label[0:trains_nums]

    val_d = frame_mfcc_data[trains_nums:val_nums+trains_nums]
    val_l = frame_label[trains_nums:val_nums+trains_nums]

    print("Beigin Training!")
    classifier =svm.SVC(C=8,kernel='rbf',gamma='auto',decision_function_shape='ovr')
    classifier.fit(trains_d,trains_l.ravel())
    print("Training is Done!")
    
    # Metrics
    print("Training accuracy is ",classifier.score(val_d,val_l.ravel()))
    val_predict = classifier.predict(val_d)

    confusion_matrix = sklearn.metrics.confusion_matrix(val_l,val_predict)
    
    if category_required==5:
        labels_name =["aa","E","ih","oh","ou"]
        plot_confusion_matrix(cm = confusion_matrix,labels_name=labels_name,title="RBF SVM 5 classification(Max),acc is {}"
     .format(round(classifier.score(val_d,val_l.ravel()),2)), figname="RBF_5_Max_second")
    elif category_required==9:
        labels_name =["CH","SS","nn","RR","aa","E","ih","oh","ou"]
        plot_confusion_matrix(cm = confusion_matrix,labels_name=labels_name,title="RBF SVM 9 classification(Max),acc is {}"
     .format(round(classifier.score(val_d,val_l.ravel()),2)), figname="RBF_9_Max_second")

    



