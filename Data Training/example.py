import os    
from explore_data import PixelShiftSound
import numpy as np
import pickle


if __name__ == "__main__":
    '''
    Attention : indice2 is bigger than indice1, both indice1 and indice 2 range from[0,7280]
    '''
    
    ps = PixelShiftSound(sample_rate=16000,frame_duration=0.016,frame_shift_duration=0.008,indice1=0,indice2=2000)
    wav_data,wav_label = ps.get_all_wav_data()
    print("Wav Frame data：",wav_data.shape)
    print("Wav Frame label：",wav_label.shape)

    
    # Optional : if you want to save the data into pickle for fast via
    data_dict = {"Data":wav_data,"Label":wav_label}
    with open("data0_2000",'wb') as f1:
        pickle.dump(data_dict,f1)
    





