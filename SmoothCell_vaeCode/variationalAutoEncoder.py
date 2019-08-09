# !/usr/bin/env python
# encoding: utf-8

"""
Wrapper for vae: variational auto-encoder for WuXiNextCode AI lab
"""

import argparse
import subprocess
import sys
from datetime import datetime
import os
import math
import numpy as np
import hpSearch
import vaeCluster

print('PID: ' + repr(os.getpid()))

# Set the TF Warning Level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ZeroOneRange(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

def hpParser(act_fnc, batch, cuda_device, early_stop, epochs, f_valid, ll_samples, min_epoch, name, output_dir,
             post_layer, reg, test_fraction, hp_samples, hp_layers, hp_dims, hp_lr, hp_dropout, hp_latent):

    out = argparse.ArgumentParser(description='hpSearch')
    out.add_argument('--act_fnc', type=str, choices=['tanh', 'relu', 'elu', 'swish', 'gated'], default=act_fnc)
    out.add_argument('--batch', type=int, metavar='N', default=batch)
    out.add_argument('--cuda_device', type=str, default=cuda_device)
    out.add_argument('--early_stop', type=int, metavar='N', default=early_stop)
    out.add_argument('--epochs', type=int, metavar='N', default=epochs)
    out.add_argument('--f_valid', type=float, metavar='N', default=f_valid)
    out.add_argument('--ll_samples', type=int, metavar='N', default=ll_samples)
    out.add_argument('--min_epoch', type=int, metavar='N', default=min_epoch)
    out.add_argument('--name', type=str, metavar='str', default=name)
    out.add_argument('--output_dir', type=str, metavar='/output_path', default=output_dir)
    out.add_argument('--post_layer', choices=['zinb', 'zi_gumbel', 'regular', 'gauss', 'nb'], default=post_layer)
    out.add_argument('--reg', type=str, choices=['kld', 'mmd', 'vamp'], default=reg)
    out.add_argument('--test_fraction', type=float, choices=[ZeroOneRange(0.0, 1.0)], default=test_fraction)
    out.add_argument('--train_data', type=str, metavar='/train_path')

    out.add_argument('--nn_dims', type=int, nargs='*', metavar='N, N, N...')
    out.add_argument('--train_idx', type=str, metavar='/path/to/train_indices.txt')
    out.add_argument('--warm_up', type=float, choices=[ZeroOneRange(0.0, 1.0)])
    out.add_argument('--pheno_graph', action='store_true')

    out.add_argument('--hp_samples', type=int, metavar='N', default=hp_samples)
    out.add_argument('--hp_layers', type=int, nargs=2, metavar='N, N', default=hp_layers)
    out.add_argument('--hp_dims', type=int, nargs='*', metavar='N, N, N...', default=hp_dims)
    out.add_argument('--hp_lr', type=float, nargs=2, metavar='N, N', default=hp_lr)
    out.add_argument('--hp_dropout', type=float, nargs=2, choices=[ZeroOneRange(0.0, 1.0)], default=hp_dropout)
    out.add_argument('--hp_latent', type=int, nargs=2, metavar='N, N', default=hp_latent)

    out.add_argument('--n_reps', type=int, metavar='N')
    out.add_argument('--code_folder', type=str, metavar='/vae')
    out.add_argument('--data_input', type=str, metavar='/raw.csv')
    out.add_argument('--reference', type=str, metavar='/reference.txt')

    return out

def clusterParser(act_fnc, batch, cuda_device, early_stop, epochs, f_valid, ll_samples, min_epoch, name,
                  output_dir, post_layer, reg, test_fraction, n_reps, dropout, latent_dim, lr):

    out = argparse.ArgumentParser(description='callVAE')
    out.add_argument('--act_fnc', type=str, choices=['tanh', 'relu', 'elu', 'swish', 'gated'], default=act_fnc)
    out.add_argument('--batch', type=int, metavar='N', default=batch)
    out.add_argument('--cuda_device', type=str, default=cuda_device)
    out.add_argument('--early_stop', type=int, metavar='N', default=early_stop)
    out.add_argument('--epochs', type=int, metavar='N', default=epochs)
    out.add_argument('--f_valid', type=float, metavar='N', default=f_valid)
    out.add_argument('--ll_samples', type=int, metavar='N', default=ll_samples)
    out.add_argument('--min_epoch', type=int, metavar='N', default=min_epoch)
    out.add_argument('--name', type=str, metavar='str', default=name)
    out.add_argument('--output_dir', type=str, metavar='/output_path', default=output_dir)
    out.add_argument('--post_layer', choices=['zinb', 'zi_gumbel', 'regular', 'gauss', 'nb'], default=post_layer)
    out.add_argument('--reg', type=str, choices=['kld', 'mmd', 'vamp'], default=reg)
    out.add_argument('--test_fraction', type=float, choices=[ZeroOneRange(0.0, 1.0)], default=test_fraction)
    out.add_argument('--train_data', type=str, metavar='/train_path')

    out.add_argument('--nn_dims', type=int, nargs='*', metavar='N, N, N...')
    out.add_argument('--train_idx', type=str, metavar='/path/to/train_indices.txt')
    out.add_argument('--warm_up', type=float, choices=[ZeroOneRange(0.0, 1.0)])
    out.add_argument('--pheno_graph', action='store_true')

    out.add_argument('--hp_samples', type=int, metavar='N')
    out.add_argument('--hp_layers', type=int, nargs=2, metavar='N, N')
    out.add_argument('--hp_dims', type=int, nargs='*', metavar='N, N, N...')
    out.add_argument('--hp_lr', type=float, nargs=2, metavar='N, N')
    out.add_argument('--hp_dropout', type=float, nargs=2, choices=[ZeroOneRange(0.0, 1.0)])
    out.add_argument('--hp_latent', type=int, nargs=2, metavar='N, N')

    out.add_argument('--n_reps', type=int, metavar='N', default=n_reps)
    out.add_argument('--code_folder', type=str, metavar='/vae')
    out.add_argument('--data_input', type=str, metavar='/raw.csv')
    out.add_argument('--reference', type=str, metavar='/reference.txt')

    out.add_argument('--dropout', type=float, choices=[ZeroOneRange(0.0, 1.0)], default=dropout)
    out.add_argument('--decoder_dims', type=int, nargs='*', metavar='N, N, N...')
    out.add_argument('--encoder_dims', type=int, nargs='*', metavar='N, N, N...')
    out.add_argument('--latent_dim', type=int, metavar='N', help='latent dimension', default=latent_dim)
    out.add_argument('--lr', type=float, metavar='N', default=lr)

    return out

def getValidLL(output_dir, baseEnd):
    # locate adequate file
    local_folders = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    search_name = [x for x in local_folders if x.endswith(baseEnd)][0]

    # collect paths, Train_LL and Valid_LL
    pathLL, validLL, trainLL, runID, runWarning = [], [], [], [], []
    with open(os.path.join(output_dir, search_name, 'all_stats_list.txt'), 'r') as f:
        # column names: Path; Train_LL; Valid_LL; All_LL; Train_Loss; Test_Loss; Train_Acc; Test_Acc; Total_Clusters; Warning
        f.readline()
        lines = f.readlines()
    for line in lines:
        run = line.strip().split('\t')
        pathLL.append(run[0])
        trainLL.append(float(run[1]))
        validLL.append(float(run[2]))
        runID.append(run[0].split('/')[-1])
        runWarning.append(int(run[9]))

    # collect all the data and sort by valid_LL
    llzip = zip(runID, trainLL, validLL, pathLL, runWarning)
    llzip_sorted = sorted(llzip, key=lambda t: t[2])

    # select the run with the lowest validLL, verifying that it is not because the posterior "collapsed"
    validLL.sort()
    trainLL.sort()
    validMedian = validLL[len(validLL) // 2]
    trainMedian = trainLL[len(trainLL) // 2]

    validMedian_50 = validMedian/100 * 50
    trainMedian_50 = trainMedian/100 * 50

    bk = True
    counter = 0
    while bk:
        cr, ct, cv, cp, rw = llzip_sorted[counter]

        current_score = cv * 100 / ct

        if validMedian_50 <= cv and trainMedian_50 <= ct and rw != -1 and current_score >= 0.50 and not math.isnan(rw):
            mainout = llzip_sorted[counter]
            bk = False
        else:
            counter += 1

    return mainout

def getHyperParameters(dir_file, minll_run):
    # column names: run; dropout; decoder_dims; encoder_dims; latent_dim; lr
    hyperpam_table = np.loadtxt(os.path.join(dir_file, 'hyperparameters.txt'), skiprows=1, delimiter='\t', dtype=str)
    minll_row = np.where(hyperpam_table[:, 0] == str(minll_run))[0][0]

    return hyperpam_table[minll_row]

def manualParameters(args, train_data):
    manual_parameters = ['--train_data', train_data]
    # numeric parameters
    if args.warm_up is not None:
        manual_parameters += ['--warm_up', str(args.warm_up)]
    if args.train_idx is not None:
        manual_parameters += ['--train_idx', str(args.train_idx)]
    # list parameters
    if args.nn_dims is not None:
        manual_parameters += ['--nn_dims']
        for val in args.nn_dims:
            manual_parameters.append(str(val))
    return manual_parameters


def vaeWrapper(args):
    print(datetime.now().strftime('%Y-%m-%d_%H:%M'))
    print('----------------------------- Initial Parameters -----------------------------\n')
    print(args)

    ### make main directories in case do not exist ###
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, 'outputData')):
        os.makedirs(os.path.join(args.output_dir, 'outputData'))

    ### store xtra arguments ###
    n_reps, code_folder, data_input = args.n_reps, args.code_folder, args.data_input
    del args.n_reps, args.code_folder, args.data_input

    ### run scProcessData.R: get normalized data in feather format ###
    print('\n\n----------------------------- Running scProcessData.R -----------------------------\n')

    cml_pd = 'Rscript ' + os.path.join(code_folder, 'scProcessData.R') + ' --output_dir ' + args.output_dir + \
             ' --data_input ' + data_input + ' --output_name ' + args.name
    print(cml_pd)
    subprocess.check_call(cml_pd, shell=True)
    print('\n' + datetime.now().strftime('%Y-%m-%d_%H:%M'))
    print('Normalization data finished!!!\n')

    # set up HPsearch parser
    print('\n----------------------------- SetUp Parameters for hpSearch -----------------------------\n')
    # set up train_data id
    train_data = os.path.join(args.output_dir, 'outputData', args.name + '.feather')
    print('\nTraining data:' + train_data + '\n')

    ### run hpSearch: selection of hyper parameters ###
    # set up parser
    hpsearch_parser = hpParser(args.act_fnc, args.batch, args.cuda_device, args.early_stop, args.epochs,
                               args.f_valid, args.ll_samples,
                               args.min_epoch, args.name, args.output_dir, args.post_layer, args.reg,
                               args.test_fraction,
                               args.hp_samples, args.hp_layers, args.hp_dims, args.hp_lr, args.hp_dropout,
                               args.hp_latent)
    # manually add parameters: do not run the extra NN for classification purposes
    manual_parameters_hp = manualParameters(args, train_data)
    manual_parameters_hp += ['--pheno_graph']

    # run hpSearch
    print('\n----------------------------- Running hpSearch.py -----------------------------\n')
    hpsearch_args = hpsearch_parser.parse_args(manual_parameters_hp)
    print(hpsearch_args)

    hpSearch.getParameters(hpsearch_args)
    print('\n' + datetime.now().strftime('%Y-%m-%d_%H:%M'))
    print('Iterations needed for hyperparameters finished!!!\n')

    # remove detailed log for HPsearch
    hp_directory = [x for x in [f for f in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, f))] if x.endswith('HPsearch_TF_VAE')][0]
    hp_rmLog_cm = 'rm ' + os.path.join(args.output_dir, hp_directory, 'runs_output.log')
    print(hp_rmLog_cm)
    subprocess.check_call(hp_rmLog_cm, shell=True)

    hpDirectory, hpBasename = args.output_dir, 'HPsearch_TF_VAE'

    ### get hyperparameters from the run with the minimum Valid_LL ###
    print('\n----------------------------- Recovering Hyperparameters -----------------------------\n')

    picked_run, picked_trainLL, picked_validLL, picked_path, warning_run = getValidLL(hpDirectory, hpBasename)
    print(os.path.dirname(picked_path), picked_run)

    run, dropout, decoder_dims, encoder_dims, latent_dim, lr = getHyperParameters(os.path.dirname(picked_path), int(picked_run.split('_')[-1]))
    print('\nHyperparameters are taken from: ' + str(picked_run))
    print('\nLearning Rate:' + str(lr))
    print('\nLatent Dimensions:' + str(latent_dim))
    print('\nDropout:' + str(dropout))
    print('\nEncoder Dimensions:' + str(encoder_dims))
    print('\nDecoder Dimensions:' + str(decoder_dims))

    print('\n' + datetime.now().strftime('%Y-%m-%d_%H:%M'))
    print('Recovering Hyperparameters finished!!!\n')

    # set up vaeCluster parser
    print('\n----------------------------- SetUp vaeCluster runs -----------------------------\n')
    ### get train_idx ###
    train_idx = os.path.join(picked_path, 'all_data_idx.txt')
    print('Training IDs are taken from: ' + train_idx + '\n')

    # set up parser
    clusearch_parser = clusterParser(args.act_fnc, args.batch, args.cuda_device, args.early_stop, args.epochs, args.f_valid, args.ll_samples,
                                     args.min_epoch, args.name, args.output_dir, args.post_layer, args.reg, args.test_fraction,
                                     n_reps, dropout, int(float(latent_dim)), float(lr))

    # manually add parameters
    manual_parameters_clu = manualParameters(args, train_data)
    manual_parameters_clu += ['--train_idx', train_idx, '--pheno_graph']

    # underscored parameters
    manual_parameters_clu += ['--decoder_dims']
    for val in decoder_dims.split('_'):
        manual_parameters_clu.append(val)
    manual_parameters_clu += ['--encoder_dims']
    for val in encoder_dims.split('_'):
        manual_parameters_clu.append(val)

    ### ----------------------------------------------------- ###
    ### run vaeCluster ###
    print('\n----------------------------- Running vaeCluster.py -----------------------------\n')
    clusearch_args = clusearch_parser.parse_args(manual_parameters_clu)
    print(clusearch_args)

    vaeCluster.vaeClustering(clusearch_args)

    print('\n' + datetime.now().strftime('%Y-%m-%d_%H:%M'))
    print('First Round of VAE clustering finished!!!\n')

    # remove detailed log for CLUSTERsearch
    cl_directory = [x for x in [f for f in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, f))] if x.endswith('CLUSTERsearch_TF_VAE')][0]
    cl_rmLog_cm = 'rm ' + os.path.join(args.output_dir, cl_directory, 'runs_output.log')
    print(cl_rmLog_cm)
    subprocess.check_call(cl_rmLog_cm, shell=True)

    ### -------------------------------------------- ###
    ### run clusteringConsensus.R: consensus cluster ###
    cml_cc = 'Rscript ' + os.path.join(code_folder, 'clusteringConsensus.R') + ' --output_dir ' + args.output_dir + ' --logNormData ' + train_data + ' --output_name ' + args.name

    print('\n----------------------------- Running clusteringConsensus.R -----------------------------\n')
    print(cml_cc)

    subprocess.check_call(cml_cc, shell=True)

    print('\n' + datetime.now().strftime('%Y-%m-%d_%H:%M'))
    print('Finished!!!\n')

def main():
    # Set up a command line execution.
    parser = argparse.ArgumentParser(description='Variational Auto-Encoder: clustering.')
    parser.add_argument('--act_fnc', type=str, required=True, choices=['tanh', 'relu', 'elu', 'swish'], help='activation function to use for layers')
    parser.add_argument('--batch', type=int, required=True, metavar='N', help='size of mini-batch')
    parser.add_argument('--cuda_device', type=str, default=None, metavar='N', help='cuda device ID to use')
    parser.add_argument('--early_stop', type=int, metavar='N', required=True, help='early stopping -- patience (default: None')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N', help='number of training epochs (default: 1000)')
    parser.add_argument('--f_valid', type=float, default=0.20, metavar='N', help='validation fraction for early stopping (default: 0.20')
    parser.add_argument('--ll_samples', default=10, type=int, metavar='N', help='number of MC samples for LL calculation')
    parser.add_argument('--min_epoch', default=0, type=int, metavar='N', help='minimum number of epochs to train before starting patience')
    parser.add_argument('--name', type=str, default='', metavar='str', help='name of run -- string to append to folder')
    parser.add_argument('--nn_dims', type=int, nargs='*', metavar='N, N, N...', help='list of nn layers widths (REQUIRED)')
    parser.add_argument('--n_reps', type=int, required=True, metavar='N', help='number of replicates of the fixed network structure')
    parser.add_argument('--output_dir', type=str, required=True, metavar='/output_path', help='writeable output path (REQUIRED)')
    parser.add_argument('--post_layer', choices=['zinb', 'zi_gumbel', 'regular', 'gauss', 'nb'], default='regular', help='which posterior layer to use')
    parser.add_argument('--reg', type=str, default='kld', choices=['kld', 'mmd'], help='choice of regularizer to use w/ the loss')
    parser.add_argument('--test_fraction', type=float, default=0.2, choices=[ZeroOneRange(0.0, 1.0)], help='test fraction to split off of clustered data')
    parser.add_argument('--train_idx', type=str, metavar='/path/to/train_indices.txt', help='fixed training indices')
    parser.add_argument('--warm_up', type=float, choices=[ZeroOneRange(0.0, 1.0)], help='warm up fraction of total epochs')

    parser.add_argument('--dropout', type=float, choices=[ZeroOneRange(0.0, 1.0)], default=1.0, help='dropout on all layers (1.0 is none 0.0 is all)')
    parser.add_argument('--decoder_dims', type=int, nargs='*', metavar='N, N, N...', help='list of decoder widths')
    parser.add_argument('--encoder_dims', type=int, nargs='*', metavar='N, N, N...', help='list of encoder widths')
    parser.add_argument('--latent_dim', type=int, metavar='N', help='latent dimension')
    parser.add_argument('--lr', type=float, default=0.001, metavar='N', help='learning rate (default: 0.001)')

    parser.add_argument('--hp_samples', type=int, metavar='N', help='do random hyperparameter optimization of N random samples')
    parser.add_argument('--hp_layers', type=int, nargs=2, metavar='N, N', help='hyper-parameter -- number of layers (range by 1) [min, max]')
    parser.add_argument('--hp_dims', type=int, nargs='*', metavar='N, N, N...', help='list of encoder widths (REQUIRED)')
    parser.add_argument('--hp_lr', type=float, nargs=2, metavar='N, N', help='hyper-parameter -- learning rate [min, max]')
    parser.add_argument('--hp_dropout', type=float, nargs=2, choices=[ZeroOneRange(0.0, 1.0)], help='dropout on all layers (1.0 is none 0.0 is all)')
    parser.add_argument('--hp_latent', type=int, nargs=2, metavar='N, N', help='hyper-parameter -- latent dimension (range by 2) [min, max]')

    parser.add_argument('--code_folder', type=str, default='.', metavar='/vae', help='directory with R and python scripts')
    parser.add_argument('--data_input', type=str, metavar='/raw.csv', help='path to/ raw data in csv format')
    parser.add_argument('--train_data', type=str, metavar='/train_path', help='path to training data -- feather format')
    args = parser.parse_args()

    # run main function
    vaeWrapper(args)

if __name__ == '__main__':
    main()
