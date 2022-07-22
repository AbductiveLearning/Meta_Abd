import asyncio
from statistics import mean, stdev

import yaml
import torch

from data import load_mnist
from eval_model import evaluate_model_monadic


def test_sum(test_file, p_model_path, imgs_data, loop, sem):

    with open(test_file, 'r') as f:
        test_data = yaml.full_load(f)

    label_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    p_model = torch.load(p_model_path)

    try:
        mae = evaluate_model_monadic(loop, sem, p_model, test_data, imgs_data,
                                     label_names, '../sum_learned.pl',
                                     '../prolog/tmp_sum_full/')
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())

    return mae


def main():
    task_name = 'mysum_full'
    data_path = '../../data/monadic/'

    test_file_5 = data_path + task_name + '_test' + '.yaml'
    test_file_10 = data_path + task_name + '_test_10' + '.yaml'
    test_file_100 = data_path + task_name + '_test_100' + '.yaml'

    # MNIST images
    _, imgs_test = load_mnist()

    # p_models = ['models/sum_full_5.pt',
    #             'models/sum_full_6.pt',
    #             'models/sum_full_7.pt',
    #             'models/sum_full_8.pt',
    #             'models/sum_full_9.pt']

    p_models = ['models/sum_full_12.pt',
                'models/sum_full_13.pt',
                'models/sum_full_14.pt',
                'models/sum_full_16.pt']

    mae5 = []
    mae10 = []
    mae100 = []

    sem = asyncio.Semaphore(16)
    loop = asyncio.get_event_loop()

    try:
        for p_model in p_models:
            print('Testing model {}:'.format(p_model))
            print('Length 5:', end=" ")
            m5 = test_sum(test_file_5, p_model, imgs_test, loop, sem)
            print(m5)
            mae5.append(m5)
            print('Length 10:', end=" ")
            m10 = test_sum(test_file_10, p_model, imgs_test, loop, sem)
            print(m10)
            mae10.append(m10)
            print('Length 100:', end=" ")
            m100 = test_sum(test_file_100, p_model, imgs_test, loop, sem)
            print(m100)
            mae100.append(m100)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

    print(mae5, mean(mae5), stdev(mae5))
    print(mae10, mean(mae10), stdev(mae10))
    print(mae100, mean(mae100), stdev(mae100))


if __name__ == "__main__":
    mae = main()
