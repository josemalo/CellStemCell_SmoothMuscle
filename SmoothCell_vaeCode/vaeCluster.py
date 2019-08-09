# !/usr/bin/env python
# encoding: utf-8

"""
vaeCluster: Run the posterior rounds of zinb-VAE to determine the number of clusters or to replicate a given number of clusters
"""

import argparse
import subprocess
import math
import os
import timeit
from datetime import datetime
import callVAE

print('PID: ' + repr(os.getpid()))

class ZeroOneRange(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end


def vaeClustering(args):
    print(args)

    # set timer
    time_start = timeit.default_timer()
    time_append = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    # Parse Output Directory
    full_path = os.path.join(args.output_dir, time_append + '_' + args.name + '_CLUSTERsearch_TF_VAE')
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    print('\nOutput directory is located at: ' + full_path + '/\n')

    # setup summary output files
    with open(os.path.join(full_path, 'pid.txt'), 'w') as fp:
        fp.write('PID: ' + repr(os.getpid()))
    with open(os.path.join(full_path, "all_stats_list.txt"), "w") as fid:
        print(*['Path', 'Train_LL', 'Valid_LL', 'All_LL', 'Train_Loss', 'Test_Loss', 'Train_Acc', 'Test_Acc', 'Total_Clusters', 'Warning'], sep='\t', file=fid)

    # flags
    warning_flag = False

    # warning to report outliers in the phenograph output or collapsed of the gradient descent
    with open(os.path.join(full_path, 'warning.txt'), 'w') as caution:
        caution.write('')

    # Remove n_reps and iterate through number of repetitions
    run_args = args
    total_iter = args.n_reps
    del run_args.n_reps
    run = 0

    while run < total_iter:
        print('Sample Number: (' + repr(run + 1) + ' of ' + repr(total_iter) + ')')
        # Make a sub-directory
        sub_path = os.path.join(full_path,  'model_run_' + repr(run+1))
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)
        # update paths
        run_args.base_path = full_path
        run_args.path = sub_path

        print(run_args)
        print('\n')
        cluster_list = callVAE.vae(run_args)

        # move to next iteration if gradient descent converged
        if not math.isnan(cluster_list[8]):
            run += 1
            # check whether there is a clustering with "single" groups
            if cluster_list[8] == -1:
                warning_flag = True
        else:
            print('\n\nGradient Descent Collapsed, a New Run With Random Hyperparameters Will Take Place.\n')

    # remove empty files
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
    parser = argparse.ArgumentParser(description='Variational Auto-Encoder: clustering.')
    parser.add_argument('--act_fnc', type=str, required=True, choices=['tanh', 'relu', 'elu', 'swish'], help='activation function to use for layers')
    parser.add_argument('--batch', type=int, required=True, metavar='N', help='size of mini-batch')
    parser.add_argument('--cuda_device', type=str, default=None, metavar='N', help='cuda device ID to use')
    parser.add_argument('--dropout', type=float, choices=[ZeroOneRange(0.0, 1.0)], default=1.0, help='dropout on all layers (1.0 is none 0.0 is all)')
    parser.add_argument('--decoder_dims', type=int, required=True, nargs='*', metavar='N, N, N...', help='list of decoder widths (REQUIRED)')
    parser.add_argument('--encoder_dims', type=int, required=True, nargs='*', metavar='N, N, N...', help='list of encoder widths (REQUIRED)')
    parser.add_argument('--early_stop', type=int, metavar='N', required=True, help='early stopping -- patience (default: None')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N', help='number of training epochs (default: 1000)')
    parser.add_argument('--f_valid', type=float, default=0.20, metavar='N', help='validation fraction for early stopping (default: 0.20')
    parser.add_argument('--latent_dim', type=int, required=True, metavar='N', help='latent dimension')
    parser.add_argument('--ll_samples', default=10, type=int, metavar='N', help='number of MC samples for LL calculation')
    parser.add_argument('--lr', type=float, default=0.001, metavar='N', help='learning rate (default: 0.001)')
    parser.add_argument('--min_epoch', default=0, type=int, metavar='N', help='minimum number of epochs to train before starting patience')
    parser.add_argument('--name', type=str, default='', metavar='str', help='name of run -- string to append to folder')
    parser.add_argument('--nn_dims', type=int, nargs='*', metavar='N, N, N...', help='list of nn layers widths (REQUIRED)')
    parser.add_argument('--n_reps', type=int, required=True, metavar='N', help='number of replicates of the fixed network structure')
    parser.add_argument('--output_dir', type=str, required=True, metavar='/output_path', help='writeable output path (REQUIRED)')
    parser.add_argument('--pheno_graph', action='store_true', help='run PhenoGraph on the latent layer after learning')
    parser.add_argument('--post_layer', choices=['zinb', 'zi_gumbel', 'regular', 'gauss', 'nb'], default='regular', help='which posterior layer to use')
    parser.add_argument('--reg', type=str, default='kld', choices=['kld', 'mmd'], help='choice of regularizer to use w/ the loss')
    parser.add_argument('--test_fraction', type=float, default=0.2, choices=[ZeroOneRange(0.0, 1.0)], help='test fraction to split off of clustered data')
    parser.add_argument('--train_data', type=str, required=True, metavar='/train_path', help='path to training data -- feather format (REQUIRED)')
    parser.add_argument('--train_idx', type=str, metavar='/path/to/train_indices.txt', help='fixed training indices')
    parser.add_argument('--warm_up', type=float, choices=[ZeroOneRange(0.0, 1.0)], help='warm up fraction of total epochs')
    args = parser.parse_args()

    # run main function
    vaeClustering(args)


if __name__ == '__main__':
    main()
