# !/usr/bin/env python
# encoding: utf-8

"""
callVAE: Run zero-inflated negative binomial Variational Auto-Encoder
"""

import argparse
import feather
import random
import os
import logging
import progressbar
import networkx as nx
import numpy as np
import pandas as pd
import math
from math import log
from math import ceil
from copy import deepcopy
from phenograph import cluster
from datetime import datetime
from sklearn.model_selection import train_test_split
from utilities import DatasetManager
from utilities import BaseVAE
from utilities import NeuralNetClassifier


# Set the TF Warning Level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ZeroOneRange(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

### vae utilities
def batch_generator(data, batch_size, shuffle=True):
    curr_idx = 0
    shuffle_idx = list(range(0, data.shape[0]))
    if shuffle:
        random.shuffle(shuffle_idx)
    # n_batches = int(floor(data.shape[0] / batch_size))
    n_batches = int(ceil(data.shape[0] / batch_size))
    n = 0
    while n < n_batches:
        begin = curr_idx
        stop = curr_idx + batch_size
        if stop > data.shape[0]:
            stop = data.shape[0]
        diff = stop - data.shape[0]
        if diff <= 0:
            batch_data = data[shuffle_idx[begin:stop],]
            # curr_idx = stop + 1
            curr_idx = stop
            n += 1
            yield batch_data

def batch_generator_labels(data, labels, batch_size, shuffle=True):
    curr_idx = 0
    shuffle_idx = list(range(0, data.shape[0]))
    if shuffle:
        random.shuffle(shuffle_idx)
    n_batches = int(ceil(data.shape[0] / batch_size))
    n = 0
    while n < n_batches:
        begin = curr_idx
        stop = curr_idx + batch_size
        if stop > data.shape[0]:
            stop = data.shape[0]
        diff = stop - data.shape[0]
        if diff <= 0:
            batch_data = data[shuffle_idx[begin:stop],]
            batch_labels = [labels[val] for val in shuffle_idx[begin:stop]]
            curr_idx = stop
            n += 1
            yield batch_data, batch_labels

def batch_generator_weights(data, weights, batch_size, shuffle=True):
    curr_idx = 0
    shuffle_idx = list(range(0, data.shape[0]))
    if shuffle:
        random.shuffle(shuffle_idx)
    n_batches = int(ceil(data.shape[0] / batch_size))
    n = 0
    while n < n_batches:
        begin = curr_idx
        stop = curr_idx + batch_size
        if stop > data.shape[0]:
            stop = data.shape[0]
        diff = stop - data.shape[0]
        if diff <= 0:
            batch_data = data[shuffle_idx[begin:stop],]
            batch_weights = weights[shuffle_idx[begin:stop],]
            curr_idx = stop
            n += 1
            yield batch_data, batch_weights

def split_data_indices(args, x_latent, communities):
    indices = np.arange(x_latent.shape[0])
    # Read in Indices and Push to NP Vector
    train_indices = pd.read_table(args.train_idx, index_col=False, delim_whitespace=False,
                                  sep='\t', header=None)
    train_indices = train_indices.as_matrix()
    train_indices = train_indices.flatten()
    test_indices = np.array([val for val in list(indices) if np.isin(val, train_indices) != True])
    # Slice a bunch of arrays by the idx
    nn_all_data = x_latent[train_indices, :]
    nn_test_data = x_latent[test_indices, :]
    nn_all_labels = communities[train_indices, ]
    nn_test_labels = communities[test_indices, ]
    nn_all_idx = train_indices
    nn_test_idx = test_indices
    # Split Train into Train/Valid for Early Stopping
    nn_train_data, nn_valid_data, nn_train_labels, nn_valid_labels = train_test_split(
        nn_all_data, nn_all_labels, test_size=args.f_valid,
        stratify=nn_all_labels)
    return (nn_all_data, nn_test_data, nn_all_labels, nn_test_labels, nn_all_idx, nn_test_idx,
            nn_train_data, nn_valid_data, nn_train_labels, nn_valid_labels)

def split_data_sk_learn(args, x_latent, communities):
    # Test Train Indices
    indices = np.arange(x_latent.shape[0])
    # Split Full Data into Train/Test
    (nn_all_data, nn_test_data, nn_all_labels,
     nn_test_labels, nn_all_idx, nn_test_idx) = train_test_split(
        x_latent, communities, indices, test_size=args.test_fraction,
        stratify=communities)
    # Split Train into Train/Valid for Early Stopping
    nn_train_data, nn_valid_data, nn_train_labels, nn_valid_labels = train_test_split(
        nn_all_data, nn_all_labels, test_size=args.f_valid,
        stratify=nn_all_labels)
    return (nn_all_data, nn_test_data, nn_all_labels, nn_test_labels, nn_all_idx, nn_test_idx,
            nn_train_data, nn_valid_data, nn_train_labels, nn_valid_labels)

def split_nn_data(args, x_latent, communities):
    if args.train_idx is not None:
        (nn_all_data, nn_test_data, nn_all_labels, nn_test_labels, nn_all_idx, nn_test_idx,
         nn_train_data, nn_valid_data, nn_train_labels, nn_valid_labels) = split_data_indices(
            args, x_latent, communities)
    else:
        (nn_all_data, nn_test_data, nn_all_labels, nn_test_labels, nn_all_idx, nn_test_idx,
         nn_train_data, nn_valid_data, nn_train_labels, nn_valid_labels) = split_data_sk_learn(
            args, x_latent, communities)

    return (nn_all_data, nn_test_data, nn_all_labels, nn_test_labels, nn_all_idx, nn_test_idx,
            nn_train_data, nn_valid_data, nn_train_labels, nn_valid_labels)

### vae sub-functions
def batch_train_nn_step(nn, batch, data, labels, epoch):
    base_logger = logging.getLogger('VAE')
    # Training by Batches
    total_cost = 0
    total_acc = 0
    num_batches = 0
    batch_gen = batch_generator_labels(data, labels, batch)
    for idx, (batch_data, batch_labels) in enumerate(batch_gen):
        cost, acc = nn.partial_fit(batch_data, batch_labels, True)
        total_cost += cost
        total_acc += np.sum(acc)
        num_batches += 1
        base_logger.info('Train Epoch ' + repr(epoch + 1) + ' - Mini-Batch ' + repr(idx + 1) + ': (Loss=' + repr(cost) + ')')
    base_logger.info('============END TRAINING BATCHES============')
    acc = total_acc / data.shape[0]
    return total_cost, num_batches, acc

def batch_output_nn_step(nn, batch, data, labels, epoch, name='VALID'):
    base_logger = logging.getLogger('VAE')
    # Validation by Batches
    valid_cost = 0
    valid_batches = 0
    valid_acc = 0
    y_list = []
    prob_list = []
    batch_valid = batch_generator_labels(data, labels, batch, shuffle=False)
    for idx, (batch_data, batch_labels) in enumerate(batch_valid):
        prob, y, correct, loss = nn.get_model_output(batch_data, batch_labels, False)
        y_list.append(y)
        prob_list.append(prob)
        valid_cost += loss
        valid_acc += np.sum(correct)
        valid_batches += 1
        base_logger.info(name + ' Epoch ' + repr(epoch + 1) + ' - Mini-Batch ' + repr(idx + 1) +
                         ': (Loss=' + repr(loss) + ')')
    base_logger.info('============END ' + name + ' BATCHES============')
    acc = valid_acc / data.shape[0]
    y_out = np.hstack(y_list)
    prob_out = np.vstack(prob_list)
    return valid_cost, valid_batches, acc, prob_out, y_out

def batch_loss_step(vae, batch, data, weights, epoch, name='VALID'):
    base_logger = logging.getLogger('VAE')
    # Validation by Batches
    batch_valid = batch_generator_weights(data, weights, batch, shuffle=False)
    valid_cost = 0
    valid_batches = 0
    for idx, (batch_data, batch_weights) in enumerate(batch_valid):
        cost, beta = vae.get_loss(batch_data, batch_weights, False, epoch)
        valid_cost += cost
        valid_batches += 1
        base_logger.info(name + ' Epoch ' + repr(epoch + 1) + ' - Mini-Batch ' + repr(idx + 1) +
                         ': (Loss=' + repr(cost) + ')')
    base_logger.info('============END ' + name + ' BATCHES============')
    return valid_cost, beta, valid_batches

def batch_train_step(vae, batch, data, weights, epoch):
    base_logger = logging.getLogger('VAE')
    # Training by Batches
    total_cost = 0
    num_batches = 0
    batch_gen = batch_generator_weights(data, weights, batch)
    for idx, (batch_data, batch_weights) in enumerate(batch_gen):
        cost, reg_loss, rec_loss = vae.partial_fit(batch_data, batch_weights, True, epoch)
        total_cost += cost
        num_batches += 1
        base_logger.info('Train Epoch ' + repr(epoch + 1) + ' - Mini-Batch ' + repr(idx + 1) +
                         ': (Total_Loss=' + repr(cost) + ') (Reg_Loss=' + repr(reg_loss) +
                         ', Rec_Loss=' + repr(rec_loss) + ')')
    base_logger.info('============END TRAINING BATCHES============')
    return total_cost, num_batches

def ll_as_numpy(vae, batch, epochs, all_data):
    base_logger = logging.getLogger('VAE')
    ll_list = []
    batch_all = batch_generator(all_data, batch, shuffle=False)
    for idx, batch_data in enumerate(batch_all):
        total_cost = vae.get_ll(batch_data, False, epochs)
        ll_list.append(total_cost)
    # Stack to numpy array
    return np.hstack(ll_list)

def calc_ll(vae, batch, epochs, ll_samples, all_data, full_path, name):
    base_logger = logging.getLogger('VAE')
    # Importance Sampling
    # Validation by Batches
    print('\nLog Likelihood Estimate -- ' + name)
    bar = progressbar.ProgressBar()

    total_cost = np.zeros((all_data.shape[0],))
    base_logger.info(name + ' LL Iteration')
    for i in bar(range(ll_samples)):
        total_cost += ll_as_numpy(vae, batch, epochs, all_data)
    # Average LL
    avg_ll = total_cost / ll_samples
    base_logger.info(name + ' Mean LL: (' + repr(np.mean(avg_ll)) + ')')
    np.savetxt(os.path.join(full_path, name + '_sample_log_likelihood.txt'), avg_ll, delimiter='\t', fmt='%s')
    return np.mean(avg_ll)

def write_latent_space(vae, batch, data, full_path):
    latent_list = []
    batch_all = batch_generator(data, batch, shuffle=False)
    # Get the Latent Representation
    for idx, batch_data in enumerate(batch_all):
        x_latent_dump = vae.transform(batch_data)
        latent_list.append(x_latent_dump)
    x_latent_dump = np.vstack(latent_list)
    # Write the Latent Vars Out
    feather.write_dataframe(pd.DataFrame(x_latent_dump), dest=full_path)
    return x_latent_dump

def train_tf_nn(args, data, valid_data, all_data, test_data, labels, valid_labels, all_labels, test_labels, full_path, n_clusters, in_dim):
    base_logger = logging.getLogger('VAE')
    nn = NeuralNetClassifier(args, n_clusters, in_dim)
    base_logger.info("Training Neural Network for Classification...")
    # Run training epochs
    best_cost = float('inf')
    best_acc = 0
    n_patience = 0
    bar = progressbar.ProgressBar()
    print('\nRunning Training Epochs...\n')
    for epoch in bar(range(0, args.epochs)):
        if args.early_stop and (n_patience >= args.early_stop):
            base_logger.info('Early Stopping -- Breaking Due to Patience...')
            break
        # Training by Batches
        total_cost, num_batches, acc = batch_train_nn_step(nn, args.batch, data, labels, epoch)
        # Validation by Batches
        valid_cost, valid_batches, valid_acc, _, _ = batch_output_nn_step(nn, args.batch, valid_data, valid_labels, epoch)
        base_logger.info('Training Epoch ' + repr(epoch + 1) + ' - Complete: (Avg. Batch Loss ' +
                         repr(total_cost / num_batches) + ', Accuracy: ' + repr(acc) + ')')
        # Log average mini batch stats
        base_logger.info('Validation Epoch ' + repr(epoch + 1) + ' - Complete: (Avg. Batch Loss ' +
                         repr(valid_cost / valid_batches) + ', Accuracy: ' + repr(valid_acc) + ')')
        # Check Patience
        if (epoch+1) > args.min_epoch:
            if (valid_cost < best_cost) and (valid_acc >= best_acc):
                nn.save_model(full_path)
                best_cost = valid_cost
                best_acc = valid_acc
                n_patience = 0
            else:
                n_patience += 1
                base_logger.info('Early Stopping -- Increasing Patience... : ' + repr(n_patience))
    # Reload Best Model
    nn.restore_model(full_path)
    # Training Data
    all_cost, all_batches, all_acc, all_prob, all_y = batch_output_nn_step(
        nn, args.batch, all_data, all_labels, args.epochs, 'TRAIN + VALID')
    # Log average mini batch stats
    base_logger.info('Final Epoch -- All Data ' + ' - Complete: (Avg. Batch Loss ' +
                     repr(all_cost / all_batches) + ', Accuracy: ' + repr(all_acc) + ')')
    # Valid Data
    test_cost, test_batches, test_acc, test_prob, test_y = batch_output_nn_step(
        nn, args.batch, test_data, test_labels, args.epochs, 'TEST')
    # Log average mini batch stats
    base_logger.info('Final Epoch -- Test Data ' + ' - Complete: (Avg. Batch Loss ' +
                     repr(test_cost / test_batches) + ', Accuracy: ' + repr(test_acc) + ')')

    # Write out Final Predictions from NN
    np.savetxt(os.path.join(full_path, 'all_predicted_labels.txt'), all_y, delimiter='\t', fmt='%s')
    np.savetxt(os.path.join(full_path, 'test_predicted_labels.txt'), test_y, delimiter='\t', fmt='%s')
    # Write out prediction probabilities
    np.savetxt(os.path.join(full_path, 'all_probabilities.txt'), all_prob, delimiter='\t', fmt='%s')
    np.savetxt(os.path.join(full_path, 'test_probabilities.txt'), test_prob, delimiter='\t', fmt='%s')
    # Write actual labels
    np.savetxt(os.path.join(full_path, 'all_actual_labels.txt'), all_labels, delimiter='\t', fmt='%s')
    np.savetxt(os.path.join(full_path, 'test_actual_labels.txt'), test_labels, delimiter='\t', fmt='%s')
    return nn, all_cost, test_cost, all_acc, test_acc, all_y

def train_tf_vae(args, data, full_path, v_name):
    base_logger = logging.getLogger('VAE')
    base_logger.info('Training VAE: ' + v_name)
    # Create a VAE
    vae = BaseVAE(args, data.train_data.shape[1], v_name)

    # flag to report if the gradient descent process was convergent
    gradient_convergent = True

    # Run training epochs
    best_cost = float('inf')
    best_train_cost = float('inf')
    n_patience = 0
    bar = progressbar.ProgressBar()
    print('\nRunning Training Epochs...\n')
    for epoch in bar(range(0, args.epochs)):
        if args.early_stop and (n_patience >= args.early_stop):
            base_logger.info('Early Stopping -- Breaking Due to Patience...')
            break
        # Training by Batches
        total_cost, num_batches = batch_train_step(vae, args.batch, data.train_scaled_data,
                                                   data.train_weights, epoch)

        # Validation by Batches
        valid_cost, beta, valid_batches = batch_loss_step(vae, args.batch, data.valid_scaled_data,
                                                          data.valid_weights, epoch)

        # check whether the cost is well defined
        if math.isnan(total_cost) or math.isnan(valid_cost):
            print('\nGradient Descent Collapsed, a New Run Will Be Executed\n')
            base_logger.info('Gradient Descent Collapsed ...')
            gradient_convergent = False
            break
        else:
            base_logger.info('Warm-Up: (Beta=' + repr(beta) + ')')
            base_logger.info('Training Epoch ' + repr(epoch + 1) + ' - Complete: (Avg. Batch Loss ' +
                             repr(total_cost / num_batches) + ')')
            # Log average mini batch stats
            base_logger.info('Validation Epoch ' + repr(epoch + 1) + ' - Complete: (Avg. Batch Loss ' +
                             repr(valid_cost / valid_batches) + ')')

            base_logger.info('COMPUTED VALUES: total_cost and valid_cost:')
            base_logger.info((total_cost, valid_cost))

            # Check Patience
            if (epoch+1) > args.min_epoch:
                if (valid_cost < best_cost) and (total_cost <= best_train_cost):
                    vae.save_model(full_path)
                    best_cost = valid_cost
                    best_train_cost = total_cost
                    n_patience = 0
                else:
                    n_patience += 1
                    base_logger.info('Early Stopping -- Increasing Patience... : ' + repr(n_patience))

    if gradient_convergent:
        # Reload Best Model
        vae.restore_model(full_path)
        # Training Data
        all_cost, _, all_batches = batch_loss_step(vae, args.batch, data.train_scaled_data, data.train_weights, args.epochs, 'TRAIN')
        # Log average mini batch stats
        base_logger.info('Loaded Epoch -- Train Data ' + ' - Complete: (Avg. Batch Loss ' + repr(all_cost / all_batches) + ')')
        # Valid Data
        all_cost, _, all_batches = batch_loss_step(vae, args.batch, data.valid_scaled_data, data.valid_weights, args.epochs, 'VALID')
        # Log average mini batch stats
        base_logger.info('Loaded Epoch -- Valid Data ' + ' - Complete: (Avg. Batch Loss ' + repr(all_cost / all_batches) + ')')

    return vae, gradient_convergent


def vae(args):
    # set up main path
    if args.path:
        full_path = args.path
    else:
        time_append = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        full_path = args.output_dir + time_append + '_' + args.name + '_TF_VAE'

    # (re-)setup the base logger
    log_run_name = 'VAE'
    base_logger = logging.getLogger(log_run_name)
    base_logger.setLevel(logging.INFO)

    fh = logging.FileHandler(os.path.join(os.path.dirname(full_path), 'runs_output.log'))
    formatter = logging.Formatter('%(asctime)s - %(name)s %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    base_logger.addHandler(fh)
    base_logger.info('Spinning Up TF VAE...')
    base_logger.info('Arguments: ' + repr(args))

    # Set CUDA Device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

    # create directories
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    full_path_vae = os.path.join(full_path, 'vae_model')
    if not os.path.exists(full_path_vae):
        os.makedirs(full_path_vae)
    full_path_nn = os.path.join(full_path, 'nn_model')
    if not os.path.exists(full_path_nn):
        os.makedirs(full_path_nn)

    # Training Data
    train_data = DatasetManager(args.f_valid, args.post_layer, args.train_data)

    # Variational Auto Encoder
    # Train the VAE
    vae, gradient_convergent = train_tf_vae(args, train_data, full_path_vae, 'base')

    if gradient_convergent:
        # Write the Final Latent Space
        x_latent = write_latent_space(vae, args.batch, train_data.all_scaled_data, os.path.join(full_path_vae, 'x_latent_final.feather'))
        # Calculate the Final LL
        train_ll = calc_ll(vae, args.batch, args.epochs, args.ll_samples, train_data.train_scaled_data, full_path_vae, 'train')
        valid_ll = calc_ll(vae, args.batch, args.epochs, args.ll_samples, train_data.valid_scaled_data, full_path_vae, 'valid')
        all_ll = calc_ll(vae, args.batch, args.epochs, args.ll_samples, train_data.all_scaled_data, full_path_vae, 'all')
        return_list = [train_ll, valid_ll, all_ll, None, None, None, None, None, 0]

        # Run phenograph step, according to the parameters
        outlier_flag = False
        if args.pheno_graph:
            print('\n ---PhenoGraph---\n')
            communities, sparse_graph, q = cluster(x_latent)

            # catch outliers
            if -1 in communities:
                outlier_flag = True
                np.savetxt(os.path.join(full_path_vae, 'phenograph_cluster_with_outliers.txt'), communities,
                           delimiter='\t', fmt='%s')
                # create sporious cluster out of the outliers
                communities[communities == -1] = max(communities) + 1
                with open(os.path.join(args.base_path, 'warning.txt'), 'a') as caution:
                    caution.write(full_path + ': Phenograph Clustering Contains Outliers\n')
                return_list[8] = -1

            np.savetxt(os.path.join(full_path_vae, 'phenograph_cluster.txt'), communities, delimiter='\t', fmt='%s')
            nx_g = nx.from_scipy_sparse_matrix(sparse_graph)
            nx.write_weighted_edgelist(nx_g, os.path.join(full_path_vae, 'nx_graph_edge_weight_list.txt'),
                                       delimiter='\t')
        else:
            communities = None

        # NN for classification
        if outlier_flag:
            return_list[3] = 0
            return_list[4] = 0
            return_list[5] = 0
            return_list[6] = 0
            return_list[7] = np.unique(communities).shape[0]
        else:
            print('\n ---Extra NN---\n')
            base_logger.info('Post-Clustering NN w/ ' + repr(np.unique(communities).shape[0]) + ' Clusters...')
            # Split up the Data for NN Training
            (nn_all_data, nn_test_data, nn_all_labels, nn_test_labels,
             nn_all_idx, nn_test_idx, nn_train_data, nn_valid_data,
             nn_train_labels, nn_valid_labels) = split_nn_data(args, x_latent, communities)
            # Train NN
            nn, all_cost, test_cost, all_acc, test_acc, all_y = train_tf_nn(
                args, nn_train_data, nn_valid_data, nn_all_data, nn_test_data, nn_train_labels,
                nn_valid_labels, nn_all_labels, nn_test_labels, full_path_nn,
                np.unique(communities).shape[0], x_latent.shape[1])
            # Return List in Order
            return_list[3] = all_cost
            return_list[4] = test_cost
            return_list[5] = all_acc
            return_list[6] = test_acc
            return_list[7] = np.unique(communities).shape[0]
            # Write IDX from Splitting
            np.savetxt(os.path.join(full_path, 'all_data_idx.txt'), nn_all_idx, delimiter='\t', fmt='%s')
            np.savetxt(os.path.join(full_path, 'test_data_idx.txt'), nn_test_idx, delimiter='\t', fmt='%s')

        if args.base_path:
            with open(os.path.join(args.base_path, 'all_stats_list.txt'), "a") as fid:
                outline = [full_path] + return_list
                print(*outline, sep='\t', file=fid)
    else:
        return_list = [0, 0, 0, 0, 0, 0, 0, 0, float('nan')]

    return return_list

def main():
    # Set up a command line execution.
    parser = argparse.ArgumentParser(description='Execute a zer-inflated negative binomial auto-endoer through TensorFlow.')
    parser.add_argument('--act_fnc', type=str, required=True, choices=['tanh', 'relu', 'elu', 'swish', 'gated'], help='activation function to use for layers')
    parser.add_argument('--batch', type=int, required=True, metavar='N', help='size of mini-batch')
    parser.add_argument('--base_path', type=str, metavar='str', help='Existing base path to write into')
    parser.add_argument('--cuda_device', default='', type=str, help='device ID to use as GPU')
    parser.add_argument('--dropout', type=float, choices=[ZeroOneRange(0.0, 1.0)], default=1.0, help='dropout on all layers (1.0 is none 0.0 is all)')
    parser.add_argument('--decoder_dims', type=int, required=True, nargs='*', metavar='N, N, N...', help='list of decoder widths (REQUIRED)')
    parser.add_argument('--encoder_dims', type=int, required=True, nargs='*', metavar='N, N, N...', help='list of encoder widths (REQUIRED)')
    parser.add_argument('--early_stop', type=int, metavar='N', help='early stopping -- patience (default: None')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N', help='number of training epochs (default: 1000)')
    parser.add_argument('--f_valid', type=float, default=0.20, metavar='N', help='validation fraction for early stopping (default: 0.20')
    parser.add_argument('--latent_dim', type=int, required=True, metavar='N', help='latent dimension')
    parser.add_argument('--ll_samples', default=10, type=int, metavar='N', help='number of MC samples for LL calculation')
    parser.add_argument('--lr', type=float, default=0.001, metavar='N', help='learning rate (default: 0.001)')
    parser.add_argument('--min_epoch', default=0, type=int, metavar='N', help='minimum number of epochs to train before starting patience')
    parser.add_argument('--name', type=str, default='', metavar='str', help='name of run -- string to append to folder')
    parser.add_argument('--nn_dims', type=int, nargs='*', metavar='N, N, N...', help='list of nn layers widths (REQUIRED)')
    parser.add_argument('--output_dir', type=str, required=True, metavar='/output_path', help='writeable output path (REQUIRED)')
    parser.add_argument('--path', type=str, metavar='str', help='Existing path to write into')
    parser.add_argument('--pheno_graph', action='store_true', help='run PhenoGraph on the latent layer after learning')
    parser.add_argument('--post_layer', choices=['zinb', 'zi_gumbel', 'regular', 'gauss', 'nb'], default='zinb', help='which posterior layer to use')
    parser.add_argument('--reg', type=str, default='kld', choices=['kld', 'mmd', 'vamp'], help='choice of regularizer to use w/ the loss')
    parser.add_argument('--test_fraction', type=float, default=0.2, choices=[ZeroOneRange(0.0, 1.0)], help='test fraction to split off of clustered data')
    parser.add_argument('--train_data', type=str, required=True, metavar='/train_path', help='path to training data -- feather format (REQUIRED)')
    parser.add_argument('--train_idx', type=str, metavar='/path/to/train_indices.txt', help='fixed training indices')
    parser.add_argument('--warm_up', type=float, choices=[ZeroOneRange(0.0, 1.0)], help='warm up fraction of total epochs')
    args = parser.parse_args()

    vae(args)


if __name__ == '__main__':
    main()
