import numpy as np
import torch
import torchaudio

# This function generates an array of zero where there are random ones start from random position and goes till width
def zerosArray(size,k,width=1):
    new_arr = np.zeros((size))
    random_numbers = np.random.permutation(size)[:k]
    for j in random_numbers:
        for k in range(j,j+width):
            # Condition to check for out of bound
            if(k < size):
                new_arr[k] = 1
    return torch.tensor(new_arr)      

# This function generates an array of zero where there are random ones start from random position and goes till width
def onesArray(size,k,width=1):
    new_arr = np.ones((size))
    #random_numbers = np.random.choice(shape[0],k, replace=False)
    random_numbers = np.random.permutation(size)[:k]
    for j in random_numbers:
        for k in range(j,j+width):
            # Condition to check for out of bound
            if(k<size):
                new_arr[k] = 0

    return torch.tensor(new_arr)  
       

