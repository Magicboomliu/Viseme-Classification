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
import pickle

'''可视化： 混淆矩阵'''
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

'''预加重'''
# 首先对数据进行预加重
def pre_emphasis_func(signal):
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    return  emphasized_signal
'''窗口化'''
# 让每一帧的2边平滑衰减，这样可以降低后续傅里叶变换后旁瓣的强度，取得更高质量的频谱。
def Windowing(frames,frame_length):
    hamming = np.hamming(frame_length)
    # hamming = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(0, frame_length) / (frame_length - 1))
    windowed_frames =frames*hamming
    return  windowed_frames
''' 傅里叶变换'''
# 对每一帧的信号，进行快速傅里叶变换，对于每一帧的加窗信号，进行N点FFT变换，也称短时傅里叶变换（STFT），N通常取256或512，然后用如下的公式计算能量谱
def FFT(frames,NFFT):
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))
    print(pow_frames.shape)
    return  pow_frames

'''fank特征 40 dim'''
def get_fBank(powd_frames,sameple_rate,NFFT,nfilt):
    '''
    :param frames: Frames after NFFT
    :param sameple_rate: 采样率
    :param nift: 规定有多少个mel滤波器 
    :return: FBank Features
    '''
    ''' 规定mel值的上限和下限'''
    low_freq_mel = 0
    # 根据葵姐斯特采样定理可得
    high_freq_mel = 2595 * np.log10(1 + (sameple_rate / 2) / 700)
    # 所有的mel中心点，为了方便后面计算mel滤波器组，左右两边各补一个中心点
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    # 各个mel滤波器在能量谱对应点的取值
    fbank = np.zeros((nfilt, int(NFFT / 2 + 1)))
    # 各个mel滤波器中心点对应FFT的区域编码，找到有值的位置
    bin = (hz_points / (sameple_rate / 2)) * (NFFT / 2)
    for i in range(1, nfilt + 1):
        left = int(bin[i - 1])
        center = int(bin[i])
        right = int(bin[i + 1])
        for j in range(left, center):
            fbank[i - 1, j + 1] = (j + 1 - bin[i - 1]) / (bin[i] - bin[i - 1])
        for j in range(center, right):
            fbank[i - 1, j + 1] = (bin[i + 1] - (j + 1)) / (bin[i + 1] - bin[i])
    filter_banks = np.dot(powd_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)  # dB
    print(filter_banks.shape)
    return fbank,filter_banks

''' 获取MFCC特征'''
def get_mfcc_features(num_ceps,filter_banks,lifted=False,cep_lifter=23):
    # 使用DCT，提取2-13维，得到MFCC特征
    num_ceps = 13
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:(num_ceps + 1)]
    if (lifted):
        # 对Mfcc进行升弦，平滑这个特征
        cep_lifter = 23
        (nframes, ncoeff) = mfcc.shape
        n = np.arange(ncoeff)
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        mfcc *= lift
        
    return mfcc
'''Unpickle Data '''
def unpickle(file):
    with open(file, 'rb') as fo:
        dct = pickle.load(fo, encoding='bytes')
    return dct

if __name__ == "__main__":
    total_wav_list=[]
    total_label_list=[]

    data_pickle_files=["data0_2000"]  # Please filled in the Pickle files, generated from example.py
    for kkk in range(len(data_pickle_files)):
        datas =unpickle(data_pickle_files[kkk])
        wav_data = datas["Data"]
        wav_label = datas["Label"]
        print(wav_data.shape)
        print(wav_label.shape)
        useful_indexs = []
        useful_labels = []
        for index,vector in enumerate(wav_label):
            lst = vector.tolist()
            max_value = max(lst)
            max_index = lst.index(max_value)
            if max_index == 0:
                continue
            elif lst[max_index]<0.75:
                continue
            else:
                useful_indexs.append(index)
                useful_labels.append(max_index-1)
        
        '''Get useful information'''
        useful_nums= len(useful_indexs)
        wav_dim  = wav_data.shape[1]
        print(len(useful_indexs)) # Userful information index
        print(len(useful_labels))  # Useful Information  values
        print(len(set(useful_labels))) # Valid whether there is 14

        '''根据index提取有用的index的MFCC标签'''
        useful_wav = np.zeros((useful_nums,wav_dim))
        for i, index in enumerate(useful_indexs):
            useful_wav[i] = wav_data[index]
        
        useful_wav_label = np.array(useful_labels)
        print("useful wav data set: ",useful_wav.shape)
        print("useful wav label set", useful_wav_label.shape)

        for i in range(len(useful_wav)):
        # 预加重和窗口化处理
            useful_wav[i] = pre_emphasis_func(useful_wav[i])
        
        useful_wav = Windowing(useful_wav,len(useful_wav[0]))
        fft_data = FFT(useful_wav,512)
        print("Wav Frame data After FFT：",fft_data.shape)
        fbank,filter_banks=get_fBank(fft_data,16000,512,40)
        print("Wav Frame data After FBanks：",filter_banks.shape)
        mfcc_data = get_mfcc_features(num_ceps=12,filter_banks=filter_banks,
                                lifted=True)
        print("Wav Frams's MFCC features",mfcc_data.shape)
        print("Finish {}".format(kkk))
        total_wav_list.append(mfcc_data)
        total_label_list.append(useful_wav_label)

    ''' Merge them together'''
    useful_wav_data_total = total_wav_list[0]
    for ii,arrary in  enumerate(total_wav_list):
        if ii==0:
            continue
        useful_wav_data_array = arrary
        useful_wav_data_total = np.vstack((useful_wav_data_total,useful_wav_data_array))
    
    w=[]
    for marray in total_label_list:
        w.extend(marray.tolist())
    
    useful_label_data_total = np.array(w)

    print("MFCC Total is ", useful_wav_data_total.shape)
    print("Label Total is ", useful_label_data_total.shape)

    # Shuffleing
    # 首先对数据进行Normalization
    trainig_data = preprocessing.scale(useful_wav_data_total)
    
    # 对数据进行shuffer操作，混乱化
    state = np.random.get_state()
    np.random.shuffle(trainig_data)
    np.random.set_state(state)
    np.random.shuffle(useful_label_data_total)
    
    # Save into a pickle file
    dct={"mfcc":trainig_data,"labels":useful_label_data_total}
    with open("mfcc14",'wb') as f1:
         pickle.dump(dct,f1)
    


