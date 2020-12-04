'''  This code is use for rebalance the training_data'''

import os 
import pickle
import matplotlib.pyplot as plt 
import numpy as np

# 解析一个pickle文件
def unpickle(file):
    with open(file, 'rb') as fo:
        dct = pickle.load(fo, encoding='bytes')
    return dct

# 统计元素出现的个数
def countX(lst, x):
    count = 0
    for ele in lst:
        if (ele == x):
            count = count + 1
    return count

def rebalance_sample(pickle_file_path,categories_required=9):
    '''
    pickle_file_path:  unpickle the file generated from extract_mfcc.py
    categories_required : 9 Categories or 5 Categories, input is 9 or 5
    '''
    dct = unpickle(pickle_file_path)   # unpickle the file generated from extract_mfcc.py
    trainig_data = dct["mfcc"]
    trainig_data_labels = dct["labels"]

    elements= list(set(trainig_data_labels))  # Get 每个 Viseme的种类
    nums_e=[]
    for e in elements:
        nums = countX(trainig_data_labels,e)
        nums_e.append(nums)
    # print(nums_e)                          # 显示每一个Viseme的个数
    
    # 提取需要的分类的的category的数目
    save_element= elements[14-categories_required:] 
    saved_index = []
    for index in range(len(trainig_data_labels)):
        if trainig_data_labels[index] in save_element:
            saved_index.append(index)

    trainig_data_new  = np.zeros((len(saved_index),13))
    training_labels_new = []
    for ii, index in enumerate(saved_index):
        trainig_data_new[ii] = trainig_data[index]
        training_labels_new.append(trainig_data_labels[index]+categories_required-14)
    
    training_labels_new_array = np.array(training_labels_new)


    new_element = list(set(training_labels_new_array))
    nums_e=[]
    for e in new_element:
        nums = countX(training_labels_new_array,e)
        nums_e.append(nums)
  
    
    min_label_count= min(nums_e)  # 统一降采样为最小样本的数量
    trainig_data_balance = trainig_data_new[training_labels_new_array == 0][:min_label_count]
    trainig_labels_balance =  training_labels_new_array[training_labels_new_array == 0][:min_label_count]
    for i in range(categories_required):
        if i==0:
            continue
        new_indices_data = trainig_data_new[training_labels_new_array == i][:min_label_count]
        new_indices_labels = training_labels_new_array[training_labels_new_array == i][:min_label_count]
        trainig_data_balance = np.vstack((trainig_data_balance,new_indices_data))
        trainig_labels_balance = np.hstack((trainig_labels_balance,new_indices_labels))
    
    # 对数据进行shuffer操作，混乱化
    state = np.random.get_state()
    np.random.shuffle(trainig_data_balance)
    np.random.set_state(state)
    np.random.shuffle(trainig_labels_balance)

    return trainig_data_balance ,trainig_labels_balance




if __name__ == "__main__":
    trainig_data_balance ,trainig_labels_balance = rebalance_sample(pickle_file_path="mfcc14",categories_required=5)
    
    # You Can Save it into pickle , if you like
    smoteen_dict = {"mfcc":trainig_data_balance,"labels":trainig_labels_balance}
    with open("mfcc5",'wb') as f1:
        pickle.dump(smoteen_dict,f1)

        

        
    
    

    
    

   