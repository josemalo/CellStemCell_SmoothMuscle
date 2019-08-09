# !/usr/bin/env python
# encoding: utf-8

"""
hpSeach: Run the first round of zinb-VAE to set-up the parameters
"""

import argparse
import os
import random
import timeit
import subprocess
import math
import numpy as np
from copy import deepcopy
from datetime import datetime
import callVAE


print('PID: ' + repr(os.getpid()))

class ZeroOneRange(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

def hp2vae_parser(act_fnc, batch, cuda_device, epochs, early_stop, f_valid, ll_samples, min_epoch, name, output_dir, post_layer, reg, test_fraction, train_data):
    out = argparse.ArgumentParser(description='Execute a zero-inflated negative binomial auto-endoer through TensorFlow.')
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
    out.add_argument('--train_data', type=str, metavar='/train_path', default=train_data)

    out.add_argument('--base_path', type=str, metavar='str')
    out.add_argument('--path', type=str, metavar='str')

    out.add_argument('--dropout', type=float, choices=[ZeroOneRange(0.0, 1.0)], default=1.0)
    out.add_argument('--latent_dim', type=int, metavar='N')
    out.add_argument('--lr', type=float, default=0.001, metavar='N')
    out.add_argument('--encoder_dims', type=int, nargs='*', metavar='N, N, N...')
    out.add_argument('--decoder_dims', type=int, nargs='*', metavar='N, N, N...')

    out.add_argument('--dump_int', type=int, metavar='N')
    out.add_argument('--nn_dims', type=int, nargs='*', metavar='N, N, N...')
    out.add_argument('--warm_up', type=float, choices=[ZeroOneRange(0.0, 1.0)])
    out.add_argument('--pheno_graph', action='store_true')

    out.add_argument('--train_idx', type=str)
    out.add_argument('--consensus', type=str, metavar='/path/to/clusters.txt')

    return out

def setup_hyperparameters(hp_layers, hp_lr, hp_dropout, hp_latent, hp_samples, hp_dims):
    # Random Hyper-parameter Setup
    param_dict = {}
    # Number of Layers
    param_dict.update({'n_layers':  list(np.arange(hp_layers[0], hp_layers[1]+1, step=1))})
    # Learning Rate
    param_dict.update({'lr': list(np.logspace(hp_lr[0], hp_lr[1], num=5, endpoint=True))})
    param_dict.update({'dropout': list(np.linspace(hp_dropout[0], hp_dropout[1], num =5, endpoint=True))})
    param_dict.update({'latent_dim': list(np.arange(hp_latent[0], hp_latent[1]+2, step=2))})
    rnd_hyper_params = random_hyperparameter_search(hp_samples, param_dict)
    # Encoder/Decoder Lists
    encoder_list = []
    decoder_list = []
    for val in rnd_hyper_params.get('n_layers'):
        encoder_list.append(encoder_random_width(val, hp_dims))
        # Copy the Encoder and Flip it
        decoder_width = deepcopy(encoder_list[-1])
        decoder_width.reverse()
        decoder_list.append(decoder_width)
    rnd_hyper_params.update({'encoder_dims': encoder_list})
    rnd_hyper_params.update({'decoder_dims': decoder_list})
    return rnd_hyper_params

def random_hyperparameter_search(num_samples, param_dict):
    new_dict = {}
    for key, val in param_dict.items():
        low_bound = 0
        high_bound = len(val)-1
        rand_list = [val[random.randint(low_bound, high_bound)] for _ in range(0, num_samples)]
        new_dict.update({key: rand_list})
    return new_dict

def encoder_random_width(n_layers, size_list):
    new_size_list = deepcopy(size_list)
    rand_widths = []
    for i in range(0, n_layers):
        # Weighted Random List
        list_len = len(new_size_list)
        total_sum = (list_len*(list_len+1))/2
        p_list = [(i+1) / total_sum for i in range(0, len(new_size_list))]
        p_list.reverse()
        random_int = np.random.choice(list_len, p=p_list)

        rand_widths.append(new_size_list[random_int])
        new_size_list = [val for val in new_size_list if val <= rand_widths[-1]]
    return rand_widths

def getparline(hyper_dict, hypernames, r):
    outline = [str(r + 1)]
    for hn in hypernames:
        hyperpar = hyper_dict[hn][0]
        if isinstance(hyperpar, list):
            hyperpar_str = str.join('_', [str(x) for x in hyperpar])
            outline += [hyperpar_str]
        else:
            outline += ['{:f}'.format(float(hyperpar))]

    return(outline)


def getParameters(args):
    print(args)

    # set timer
    time_start = timeit.default_timer()
    time_append = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    # Parse Output Directory
    full_path = os.path.join(args.output_dir, time_append + '_' + args.name + '_HPsearch_TF_VAE')
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    print('\nOutput directory is located at: ' + full_path + '/\n')

    # warning to report outliers in the phenograph output or collapsed of the gradient descent
    with open(os.path.join(full_path, 'warning.txt'), 'w') as caution:
        caution.write('')

    # setup summary output files
    with open(os.path.join(full_path, 'pid.txt'), 'w') as fp:
        fp.write('PID: ' + repr(os.getpid()))
    with open(os.path.join(full_path, "all_stats_list.txt"), "w") as fid:
        print(*['Path', 'Train_LL', 'Valid_LL', 'All_LL', 'Train_Loss', 'Test_Loss', 'Train_Acc', 'Test_Acc', 'Total_Clusters', 'Warning'], sep = '\t', file = fid)
    # start hyperparameters file
    with open(os.path.join(full_path, 'hyperparameters.txt'), 'w') as f:
        print(*['run', 'dropout', 'decoder_dims', 'encoder_dims', 'latent_dim', 'lr'], sep='\t', file=f)
    # filed hyperparameters
    with open(os.path.join(full_path, 'FailedHyperparameters.txt'), 'w') as g:
        print(*['run', 'dropout', 'decoder_dims', 'encoder_dims', 'latent_dim', 'lr'], sep='\t', file=g)

    # Set parameters to pass-on
    input_parse = hp2vae_parser(args.act_fnc, args.batch, args.cuda_device, args.epochs, args.early_stop, args.f_valid, args.ll_samples,
                                args.min_epoch, args.name, args.output_dir, args.post_layer, args.reg, args.test_fraction, args.train_data)

    # flags
    warning_flag, nan_flag = False, False

    # Iterate through hyper-parameters (checking that the gradient descent did not collapsed) until we reach the total number of iterations
    hype_run = 0
    print('\nRunning ' + str(args.hp_samples) + ' Iterations.\n')
    while hype_run < args.hp_samples:
        print('\n\nRandom Hyperparameter Search: (' + repr(hype_run + 1) + ' of ' + repr(args.hp_samples) + ')')

        run_parse = input_parse

        # Make a sub-directory
        sub_path = os.path.join(full_path, 'hp_run_' + repr(hype_run+1))
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)

        # set-up manual parameters
        manual_parameters = ['--path', sub_path, '--base_path', full_path]
        # boolean parameters
        if args.pheno_graph:
            manual_parameters += ['--pheno_graph']

        # setup current hyperparameters
        hyper_params = setup_hyperparameters(args.hp_layers, args.hp_lr, args.hp_dropout, args.hp_latent, 1, args.hp_dims)
        print(hyper_params)

        # set-up hyperparameters
        manual_parameters += ['--dropout', str(hyper_params['dropout'][0])]
        manual_parameters += ['--latent_dim', str(hyper_params['latent_dim'][0])]
        manual_parameters += ['--lr', str(hyper_params['lr'][0])]
        manual_parameters += ['--decoder_dims']
        for val in hyper_params['decoder_dims'][0]:
            manual_parameters += [str(val)]
        manual_parameters += ['--encoder_dims']
        for val in hyper_params['encoder_dims'][0]:
            manual_parameters += [str(val)]
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

        # create parse arguments
        run_args = run_parse.parse_args(manual_parameters)

        print('\n')
        print(run_args)
        hp_list = callVAE.vae(run_args)

        # store values and move to next iteration if gradient descent converged
        if not math.isnan(hp_list[8]):
            with open(os.path.join(full_path, 'hyperparameters.txt'), 'a') as f:
                outline = getparline(hyper_params, ['dropout', 'decoder_dims', 'encoder_dims', 'latent_dim', 'lr'], hype_run)
                print(*outline, sep='\t', file=f)
            hype_run += 1
            # check whether there is a clustering with "single" groups
            if hp_list[8] == -1:
                warning_flag = True

        # otherwise store in a separate file the hyperparameters used and repeat the run with new random hyperparameters
        else:
            nan_flag = True
            print('\n\nGradient Descent Collapsed, a New Run With Random Hyperparameters Will Take Place.\n')
            with open(os.path.join(full_path, 'FailedHyperparameters.txt'), 'a') as f:
                outline = getparline(hyper_params, ['dropout', 'decoder_dims', 'encoder_dims', 'latent_dim', 'lr'], hype_run)
                print(*outline, sep='\t', file=f)

    # remove empty files
    if not nan_flag:
        cml_remove_failed = 'rm ' + os.path.join(full_path, 'FailedHyperparameters.txt')
        print('All the Gradient Descent Converged.')
        print(cml_remove_failed)
        subprocess.check_call(cml_remove_failed, shell=True)

    if not warning_flag:
        cml_remove_warning = 'rm ' + os.path.join(full_path, 'warning.txt')
        print('No Clustering Process found Single Groups.')
        print(cml_remove_warning)
        subprocess.check_call(cml_remove_warning, shell=True)

    # finish timer
    time_end = timeit.default_timer()
    with open(os.path.join(full_path, 'run_time.txt'), 'w') as f:
        f.write('Total Wall Time: ' + str((time_end - time_start) / 60) + ' min')

def main():
    # Set up a command line execution.
    parser = argparse.ArgumentParser(description='Variational Auto-Encoder: run all the iterations from which the parameters will be selected.')
    parser.add_argument('--act_fnc', type=str, required=True, choices=['tanh', 'relu', 'elu', 'swish'], help='activation function to use for layers')
    parser.add_argument('--batch', type=int, required=True, metavar='N', help='size of mini-batch')
    parser.add_argument('--cuda_device', type=str, default=None, metavar='N', help='cuda device ID to use')
    parser.add_argument('--early_stop', type=int, metavar='N', help='early stopping -- patience (default: None')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N', help='number of training epochs (default: 1000)')
    parser.add_argument('--f_valid', type=float, default=0.20, metavar='N', help='validation fraction for early stopping (default: 0.20')
    parser.add_argument('--ll_samples', default=10, type=int, metavar='N', help='number of MC samples for LL calculation')
    parser.add_argument('--min_epoch', default=0, type=int, metavar='N', help='minimum number of epochs to train before starting patience')
    parser.add_argument('--name', type=str, default='', metavar='str', help='name of run -- string to append to folder')
    parser.add_argument('--nn_dims', type=int, nargs='*', metavar='N, N, N...', help='list of nn layers widths')
    parser.add_argument('--output_dir', type=str, required=True, metavar='/output_path', help='writeable output path (REQUIRED)')
    parser.add_argument('--pheno_graph', action='store_true', help='run PhenoGraph on the latent layer after learning')
    parser.add_argument('--post_layer', default='regular', choices=['zinb', 'zi_gumbel', 'regular', 'gauss', 'nb'], help='which posterior layer to use')
    parser.add_argument('--reg', type=str, default='kld', choices=['kld', 'mmd'], help='choice of regularizer to use w/ the loss')
    parser.add_argument('--test_fraction', type=float, choices=[ZeroOneRange(0.0, 1.0)], help='test fraction to split off of clustered data')
    parser.add_argument('--train_data', type=str, required=True, metavar='/train_path', help='path to training data -- feather format (REQUIRED)')
    parser.add_argument('--train_idx', type=str, metavar='/path/to/train_indices.txt', help='fixed training indices')
    parser.add_argument('--warm_up', type=float, choices=[ZeroOneRange(0.0, 1.0)], help='warm up fraction of total epochs')

    parser.add_argument('--hp_samples', type=int, metavar='N', help='do random hyperparameter optimization of N random samples')
    parser.add_argument('--hp_layers', type=int, nargs=2, metavar='N, N', help='hyper-parameter -- number of layers (range by 1) [min, max]')
    parser.add_argument('--hp_dims', type=int, required=True, nargs='*', metavar='N, N, N...', help='list of encoder widths (REQUIRED)')
    parser.add_argument('--hp_lr', type=float, nargs=2, metavar='N, N', help='hyper-parameter -- learning rate [min, max]')
    parser.add_argument('--hp_dropout', type=float, nargs=2, choices=[ZeroOneRange(0.0, 1.0)], help='dropout on all layers (1.0 is none 0.0 is all)')
    parser.add_argument('--hp_latent', type=int, nargs=2, metavar='N, N', help='hyper-parameter -- latent dimension (range by 2) [min, max]')
    args = parser.parse_args()

    # run main function
    getParameters(args)


if __name__ == '__main__':
    main()
