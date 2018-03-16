#!/usr/bin/env python

import itertools
import subprocess


num_runs = 4
training_steps = 5000

batch_sizes = [25, 50, 100, 200]
conv1_filters = [16, 32, 64]
conv1_kernel_sizes = [3, 5, 7]
pool1_pool_sizes = [2, 3, 4]
pool1_strides = [2, 3]
conv2_filters = [16, 32, 64]
conv2_kernel_sizes = [3, 5, 7]
pool2_pool_sizes = [2, 3, 4]
pool2_strides = [2, 3]
dense_units = [128, 256, 512, 1024]
dropout_rates = [0.1, 0.3, 0.4, 0.5, 0.7]
optimizers = ['Adagrad', 'Adam', 'GradientDescent']
learning_rates = [1, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005]

all_configs = (batch_sizes, conv1_filters, conv1_kernel_sizes, pool1_pool_sizes,
               pool1_strides, conv2_filters, conv2_kernel_sizes,
               pool2_pool_sizes, pool2_strides, dense_units, dropout_rates,
               optimizers, learning_rates)
default = (100, 32, 5, 2, 2, 64, 5, 2, 2, 1024, 0.4, 'GradientDescent', 0.1)
max_diffs = 2


if __name__ == '__main__':
    configs = []
    for config in itertools.product(*all_configs):
        num_diffs = sum([x != y for (x, y) in zip(default, config)])
        if num_diffs <= max_diffs:
            configs.append(config)

    print('>>> NUM CONFIGS: %d <<<' % len(configs))
    for i in range(num_runs):
        print('>>> RUN #%d <<<' % (i + 1))
        for config in configs:
            print('>>> CONFIG: %s <<<' % str(config))
            subprocess.call(('./cnn/main.py '
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
