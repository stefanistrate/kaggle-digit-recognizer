#!/usr/bin/env python

import itertools
import subprocess


num_runs = 1
training_steps = 5000

batch_sizes = [100]
conv1_filters = [32]
conv1_kernel_sizes = [5]
pool1_pool_sizes = [2]
pool1_strides = [2]
conv2_filters = [32, 64]
conv2_kernel_sizes = [5]
pool2_pool_sizes = [2]
pool2_strides = [2]
dense_units = [512, 1024]
dropout_rates = [0.2, 0.4, 0.6]
optimizers = ['Adagrad', 'Adam', 'Ftrl', 'GradientDescent', 'RMSProp']
learning_rates = [0.25, 0.1, 0.05, 0.025]

all_configs = (batch_sizes, conv1_filters, conv1_kernel_sizes, pool1_pool_sizes,
               pool1_strides, conv2_filters, conv2_kernel_sizes,
               pool2_pool_sizes, pool2_strides, dense_units, dropout_rates,
               optimizers, learning_rates)
default = (100, 32, 5, 2, 2, 64, 5, 2, 2, 1024, 0.4, 'GradientDescent', 0.1)
max_diffs = 3


if __name__ == '__main__':
    configs = []
    for config in itertools.product(*all_configs):
        num_diffs = sum([x != y for (x, y) in zip(default, config)])
        if num_diffs <= max_diffs:
            configs.append(config)

    print('>>> NUM RUNS: %d <<<' % num_runs)
    print('>>> NUM CONFIGS: %d <<<' % len(configs))
    for i in range(num_runs):
        print('>>> RUN #%d <<<' % (i + 1))
        for config in configs:
            print('>>> CONFIG: %s <<<' % str(config))
            subprocess.call(('./cnn/deep_main.py '
                             '--training_steps {} '
                             '--batch_size {} '
                             '--conv1_filters {} '
                             '--conv1_kernel_size {} '
                             '--pool1_pool_size {} '
                             '--pool1_strides {} '
                             '--conv2_filters {} '
                             '--conv2_kernel_size {} '
                             '--pool2_pool_size {} '
                             '--pool2_strides {} '
                             '--dense_units {} '
                             '--dropout_rate {} '
                             '--optimizer {} '
                             '--learning_rate {} ').format(
                                     *((training_steps,) + config)),
                            shell=True)
