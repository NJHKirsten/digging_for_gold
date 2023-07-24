import shutil

import torch


def clear_all_experiments():
    shutil.rmtree('runs/0/trained_models')
    shutil.rmtree('runs/0/training_graphs')


def check_cuda():
    print(torch.cuda.is_available())


if __name__ == "__main__":
    check_cuda()
