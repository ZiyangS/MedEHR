import os
import random
import pickle
import numpy as np
import torch 
import scipy.io

def get_batch(CHD_data, ind, spec_num, vocab_num, device, emsize=300):
    """fetch input data by batch."""
    batch_size = len(ind)
    data_batch = np.zeros((batch_size, spec_num, vocab_num)) # D_minibatch x T x V
    for i, doc_id in enumerate(ind):
        data_batch[i]  = CHD_data[doc_id] # get the CHD data, V size vector
    data_batch = torch.from_numpy(data_batch).float().to(device)
    return data_batch

def get_label_batch(labels, ind, device):
    """fetch input labels by batch."""
    batch_size = len(ind)
    labels_batch = np.zeros(batch_size) # D_minibatch
    for i, doc_id in enumerate(ind):
        labels_batch[i] = labels[doc_id] # get the CHD data, V size vector
    labels_batch = torch.from_numpy(labels_batch).float().to(device)
    return labels_batch