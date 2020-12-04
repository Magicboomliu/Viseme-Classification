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
