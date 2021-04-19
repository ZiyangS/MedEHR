#/usr/bin/python

from __future__ import print_function

import argparse
import torch
import pickle 
import numpy as np 
import os 
import math 
import random 
import sys
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.io

import data
from sklearn.decomposition import PCA
from torch import nn, optim
from torch.nn import functional as F
from heteregenous_ETM import heteregenous_ETM
# from utils import nearest_neighbors, get_topic_coherence

parser = argparse.ArgumentParser(description='The Embedded Topic Model')

### data and file related arguments
parser.add_argument('--dataset', type=str, default='processed_toydata.pkl', help='name of dataset')
# toydata processed by bag-of-word algorithm
parser.add_argument('--data_path', type=str, default='CHDdata/', help='directory containing data')
parser.add_argument('--save_path', type=str, default='CHDresult', help='directory to save results')
parser.add_argument('--batch_size', type=int, default=300, help='number of documents in a batch for training')

### model-related arguments
parser.add_argument('--num_topics', type=int, default=100, help='number of topics')
parser.add_argument('--rho_size', type=int, default=300, help='dimension of rho')
parser.add_argument('--emb_size', type=int, default=250, help='dimension of embeddings')
parser.add_argument('--t_hidden_size', type=int, default=500, help='dimension of hidden space of q(theta)')
parser.add_argument('--theta_act', type=str, default='relu', help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)')
parser.add_argument('--train_embeddings', type=int, default=1, help='whether to fix rho or train it')

### optimization-related arguments
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--lr_factor', type=float, default=4.0, help='divide learning rate by this')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--mode', type=str, default='train', help='train or evaluate model')
parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimizer')
parser.add_argument('--seed', type=int, default=2019, help='random seed (default: 1)')
parser.add_argument('--enc_drop', type=float, default=0.0, help='dropout rate on encoder')
parser.add_argument('--eta_dropout', type=float, default=0.0, help='dropout rate on rnn for eta')
parser.add_argument('--clip', type=float, default=0.0, help='gradient clipping')
parser.add_argument('--nonmono', type=int, default=10, help='number of bad hits allowed')
parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')
parser.add_argument('--anneal_lr', type=int, default=0, help='whether to anneal the learning rate or not')
parser.add_argument('--bow_norm', type=int, default=1, help='normalize the bows or not')

### evaluation, visualization, and logging-related arguments
parser.add_argument('--num_words', type=int, default=20, help='number of words for topic viz')
parser.add_argument('--log_interval', type=int, default=10, help='when to log training')
parser.add_argument('--visualize_every', type=int, default=1, help='when to visualize results')
parser.add_argument('--eval_batch_size', type=int, default=1000, help='input batch size for evaluation')
parser.add_argument('--load_from', type=str, default='', help='the name of the ckpt to eval from')
parser.add_argument('--tc', type=int, default=0, help='whether to compute tc or not')

# whether predict together
parser.add_argument('--predict_labels', type=int, default=1, help='whether to predict labels')
parser.add_argument('--multiclass_labels', type=int, default=0, help='whether to predict labels')

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set seed
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)

# get train dataset
print('Getting dataset ...')
data_file = os.path.join(args.data_path, args.dataset)
CHD_data = pickle.load(open(data_file, "rb"))
pat_num = CHD_data.shape[0] # D, 1593
spec_num = CHD_data.shape[1] # T, 3
vocab_num = CHD_data.shape[2] # V, 3427
print(pat_num, spec_num, vocab_num)
# no validation and test right now, only train

# get label dataset
labels = pickle.load(open('CHDdata/processed_toydata_label.pkl', "rb"))
labels = torch.from_numpy(labels).float().to(device)
# get all labels
if args.predict_labels:
    args.num_labels = 1
else:
    args.num_labels = 0
print('\n')
print('=*'*100)
print('Training a heteregenous ETM Embedded Topic Model on {} with the following settings: \n{}'.format(args.dataset, args))
print('=*'*100)

## define checkpoint
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

if args.mode == 'evaluate':
    ckpt = args.load_from
else:
    ckpt = os.path.join(args.save_path,
        'heteregenous ETM {}_K_{}_Htheta_{}_Optim_{}_Clip_{}_ThetaAct_{}_Lr_{}_Bsz_{}_RhoSize_{}_trainEmbeddings_{}'.format(
        args.dataset, args.num_topics, args.t_hidden_size, args.optimizer, args.clip, args.theta_act,
            args.lr, args.batch_size, args.rho_size, args.train_embeddings))
print(ckpt)
## define model and optimizer
if args.load_from != '':
    print('Loading checkpoint from {}'.format(args.load_from))
    with open(args.load_from, 'rb') as f:
        model = torch.load(f)
else:
    model = heteregenous_ETM(args.num_topics, vocab_num, spec_num, args.t_hidden_size, args.rho_size, args.emb_size, spec_num,
                        args.theta_act, enc_drop=args.enc_drop, predict_labels=args.predict_labels,
                        multiclass_labels=args.multiclass_labels, num_labels=args.num_labels)
print('\n heteregenous ETM architecture: {}'.format(model))
model.to(device)

if args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'adadelta':
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'asgd':
    optimizer = optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
else:
    print('Defaulting to vanilla SGD')
    optimizer = optim.SGD(model.parameters(), lr=args.lr)


def train(epoch):
    """Train heteregenous ETM on data for one epoch.
    """
    model.train()
    acc_loss = 0
    acc_recon_loss = 0
    acc_kl_alpha_loss = 0
    acc_kl_theta_loss = 0
    acc_pred_loss = 0
    cnt = 0 # minibatch index
    indices = torch.randperm(pat_num)
    indices = torch.split(indices, args.batch_size)
    for idx, ind in enumerate(indices):
        optimizer.zero_grad()
        model.zero_grad()
        data_batch = data.get_batch(CHD_data, ind, spec_num, vocab_num, device) # D_minibatch x T x V
        # D_minibatch x T x V, here D_minibatch is the number of documents in the minibatch
        data_batch += 1e-12 # avoid 0/0
        sums = data_batch.sum(2).unsqueeze(2) # torch.Size([300, 3, 1]), D_minibatch x T x 1
        if args.bow_norm:
            normalized_data_batch = data_batch / sums # normalized over the words
        else:
            normalized_data_batch = data_batch

        if args.predict_labels:
            data_batch1 = data.get_batch(CHD_data, ind, spec_num, vocab_num, device)
            data_batch1 += 1e-12  # avoid 0/0
            sums1 = data_batch1.sum(2).unsqueeze(2)  # torch.Size([300, 3, 1]), D_minibatch x T x 1
            if args.bow_norm:
                normalized_data_batch1 = data_batch1 / sums1  # normalized over the words
            else:
                normalized_data_batch1 = data_batch1
            data_batch = (data_batch, data_batch1)
            normalized_data_batch = (normalized_data_batch, normalized_data_batch1)
            labels_batch = data.get_label_batch(labels, ind, device)
            loss, recon_loss, kl_alpha, kl_theta, pred_loss = model(data_batch, normalized_data_batch, pat_num, labels_batch,
                                                                    epoch)
            loss.backward()
        else:
            loss, recon_loss, kl_alpha, kl_theta = model(data_batch, normalized_data_batch, pat_num)
            loss.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        acc_loss += torch.sum(loss).item() # total loss, the objective we want to optimize
        acc_recon_loss += torch.sum(recon_loss).item()
        acc_kl_theta_loss += torch.sum(kl_theta).item()
        acc_kl_alpha_loss += torch.sum(kl_alpha).item()
        acc_pred_loss += torch.sum(pred_loss).item()
        cnt += 1

        if idx % args.log_interval == 0 and idx > 0:
            cur_loss = round(acc_loss / cnt, 2) 
            cur_recon_loss = round(acc_recon_loss / cnt, 2)
            cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
            cur_kl_alpha = round(acc_kl_alpha_loss / cnt, 2)
            cur_pred_loss = round(acc_pred_loss / cnt, 2)
            lr = optimizer.param_groups[0]['lr']
            print('Epoch: {} .. batch: {}/{} .. LR: {} .. KL_theta: {} .. KL_alpha: {} .. Prediction_loss: {} .. Reconstruction_loss: {} .. NELBO: {}'.format(
                epoch, idx, len(indices), lr, cur_kl_theta, cur_kl_alpha, cur_pred_loss, cur_recon_loss, cur_loss))
    
    cur_loss = round(acc_loss / cnt, 2) 
    cur_recon_loss = round(acc_recon_loss / cnt, 2)
    cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
    cur_kl_alpha = round(acc_kl_alpha_loss / cnt, 2)
    cur_pred_loss = round(acc_pred_loss / cnt, 2)
    lr = optimizer.param_groups[0]['lr']
    print('*'*100)
    print('Epoch----->{} .. LR: {} .. KL_theta: {} .. KL_alpha: {} .. Prediction_loss: {} .. Reconstruction_loss: {} .. NELBO: {}'.format(
            epoch, lr, cur_kl_theta, cur_kl_alpha, cur_pred_loss, cur_recon_loss, cur_loss))
    print('*'*100)


def get_theta(eta, bows):
    model.eval()
    with torch.no_grad():
        inp = torch.cat([bows, eta], dim=1)
        q_theta = model.q_theta(inp)
        mu_theta = model.mu_q_theta(q_theta)
        theta = F.softmax(mu_theta, dim=-1)
        return theta    


if args.mode == 'train':
    ## train model on data by looping through multiple epochs
    best_epoch = 0
    best_val_ppl = 1e9
    all_val_ppls = []
    for epoch in range(1, args.epochs):
        print("epoch: %d" % epoch)
        train(epoch)
        # if epoch > 100:
        if epoch % 1 == 0:
            # current_model_name = ckpt + "_epoch" + str(epoch)
            current_model_name = "CHDresult/epoch" + str(epoch ) + ".pt"
            with open(current_model_name, 'wb') as f:
                torch.save(model, f)

            beta = model.get_beta(model.mu_q_alpha).cpu().detach().numpy()
            with open('CHDresult/beta/beta_epoch' + str(epoch) + '.pkl', 'wb') as f:
                pickle.dump(beta, f)
            # scipy.io.savemat('CHDresult/beta/beta_epoch{}.mat'.format(epoch), {'values': beta}, do_compression=True)]

            if args.predict_labels:
                print('saving classifer weights...')
                classifer_weights = model.classifier.weight.cpu().detach().numpy()
                scipy.io.savemat('CHDresult/classifer/classifer_epoch{}.mat'.format(epoch), {'values': classifer_weights},
                                 do_compression=True)
    # with open(ckpt, 'rb') as f:
    #     model = torch.load(f)
    # model = model.to(device)
    # model.eval()




