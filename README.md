# Viseme-Classification
A pipeline from Dataset Gathering,Data annotations, Model training,Model Evaluation for viseme (visual sound phoneme)  classification  

### How Can I training the data if I have the wav files and Corresponding Viseme tags？  
*If you want to train it in Python , Make sure you have following site packages*:  
* numpy  
* sklearn  
* librosa  
* tensorflow(Optional)  
* keras(Optinal)  


For this question, you can see the simple of the `'Data Triaining'` File.   
This is the Tree Structure of the Data Training File
```
├── DataSet
│   ├── label
│   └── wav_data
├── dnn.py
├── example.py
├── explore_data.py
├── extract_mfcc.py
├── log_dir
├── mfcc14
├── mfcc5
├── mfcc9
├── model
├── Pipelines.py
├── __pycache__
│   └── explore_data.cpython-36.pyc
├── rebalance_samples.py
└── svm.py
```

In this directory, DataSet contains the wav_data and the label_data labelled by Oculus OVRlipSync([Oculus OVRlipSync reference](https://developer.oculus.com/documentation/native/audio-ovrlipsync-native/))
| folder | files contains |
| ------ | ------ |
| wav_data | 7820 wav files |
| label | 7820 label txt files |  

#### More about this dataset:  
 * There are 7280 wav files and their corresponding visme labels. (Data source : AISHELL dataset).  
 * All wav files are 16 bit, with a 16KHz Sample Rate, The channel is Mono( Only 1 channel).  
 * All the wav file have been spilt into frames, the frame length is 16ms，and the frame shift(which means frame's sample step) is 8ms .
 * There is a Python Interface for this datatset for fast via named `'explore_data.py'` 。 

*Example for using the `'explore_data.py'`*:  
```
import os    
from explore_data import PixelShiftSound
import numpy as np
if __name__ == "__main__":
    '''
    Attention : indice2 is bigger than indice1, both indice1 and indice 2 range from[0,7280] means how many files you want to use in Training
    '''
    
    ps = PixelShiftSound(sample_rate=16000,frame_duration=0.016,frame_shift_duration=0.008,indice1=0,indice2=2000)
    wav_data,wav_label = ps.get_all_wav_data()
    print("Wav Frame data：",wav_data.shape)
    print("Wav Frame label：",wav_label.shape)
```


 
 #### Here is the recommend Way to Training Your Data : Run the Pipeline.py  
 ```
import os

if __name__ == "__main__":
    # Get the data, save in into pickle(in case of the file is to big)
    os.system("python example.py")
    # Extract mfcc (librosa is requried), save in into pickle
    os.system("extract_mfcc.py")
    # Reblance the samples, save it into pickle
    os.system("rebalance_samples.py")
    # Run SVM(Sklearn is requried)
    os.system("svm.py")
    # Run DNN (tensorflow and Keras are required)
    os.system("dnn.py") # Optional
```
![](https://github.com/Magicboomliu/Viseme-Classification/blob/main/00000.png)    

The reason I use a pipeline is the dataset is to big for memory, it is better to do operations seperately to save times.  



* #### STEP ONE: Get Wav Frame Data and labels Data
