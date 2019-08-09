# !/usr/bin/env python
# encoding: utf-8

"""
Wrapper for clustering process vae
"""

import argparse
import subprocess
import sys
from datetime import datetime
import os
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

    out.add_argument('--n_reps', type=int, metavar='N', default=n_reps)
    out.add_argument('--code_folder', type=str, metavar='/vae')

    out.add_argument('--dropout', type=float, choices=[ZeroOneRange(0.0, 1.0)], default=dropout)
    out.add_argument('--decoder_dims', type=int, nargs='*', metavar='N, N, N...')
    out.add_argument('--encoder_dims', type=int, nargs='*', metavar='N, N, N...')
    out.add_argument('--latent_dim', type=int, metavar='N', help='latent dimension', default=latent_dim)
    out.add_argument('--lr', type=float, metavar='N', default=lr)

    return out

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

def clusterWrapper(args):
    print(datetime.now().strftime('%Y-%m-%d_%H:%M'))
    print('----------------------------- Initial Parameters -----------------------------\n')
    print(args)

    ### make main directories in case do not exist ###
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, 'outputData')):
        os.makedirs(os.path.join(args.output_dir, 'outputData'))

    ### store xtra arguments ###
    code_folder = args.code_folder
    del args.code_folder

    # set up vaeCluster parser
    print('\n----------------------------- SetUp vaeCluster runs -----------------------------\n')
    # set up parser
    clusearch_parser = clusterParser(args.act_fnc, args.batch, args.cuda_device, args.early_stop, args.epochs, args.f_valid, args.ll_samples,
                                     args.min_epoch, args.name, args.output_dir, args.post_layer, args.reg, args.test_fraction,
                                     args.n_reps, args.dropout, args.latent_dim, args.lr)

    # manually add parameters
    manual_parameters_clu = manualParameters(args, args.train_data)
    manual_parameters_clu += ['--train_idx', args.train_idx, '--pheno_graph']

    # underscored parameters
    manual_parameters_clu += ['--decoder_dims']
    for val in args.decoder_dims:
        manual_parameters_clu.append(str(val))
    manual_parameters_clu += ['--encoder_dims']
    for val in args.encoder_dims:
        manual_parameters_clu.append(str(val))

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
    parser.add_argument('--warm_up', type=float, choices=[ZeroOneRange(0.0, 1.0)], help='warm up fraction of total epochs')

    parser.add_argument('--dropout', type=float, choices=[ZeroOneRange(0.0, 1.0)], default=1.0, help='dropout on all layers (1.0 is none 0.0 is all)')
    parser.add_argument('--decoder_dims', type=int, nargs='*', metavar='N, N, N...', help='list of decoder widths')
    parser.add_argument('--encoder_dims', type=int, nargs='*', metavar='N, N, N...', help='list of encoder widths')
    parser.add_argument('--latent_dim', type=int, metavar='N', help='latent dimension')
    parser.add_argument('--lr', type=float, default=0.001, metavar='N', help='learning rate (default: 0.001)')

    parser.add_argument('--code_folder', type=str, default='.', metavar='/vae', help='directory with R and python scripts')
    parser.add_argument('--train_idx', type=str, metavar='/path/to/train_indices.txt', help='fixed training indices')
    parser.add_argument('--train_data', type=str, metavar='/train_path', help='path to training data -- feather format')
    args = parser.parse_args()

    # run main function
    clusterWrapper(args)

if __name__ == '__main__':
    main()
