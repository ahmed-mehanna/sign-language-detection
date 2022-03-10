import numpy as np

actions = ['sister','hurry','hungry','meal','brother','tree','heavy','cry','family','wise']




def softmax(x):    
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


def arg_max(array):
    arg_max = np.argmax(array)
    return arg_max,array[arg_max]


