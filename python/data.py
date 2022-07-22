from random import choice, randint, sample
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms

'''
Class of one example
'''


class Example:
    x = None  # input 1
    y = None  # input 2
    z = None  # output
    x_digits = []  # digits
    y_digits = []
    x_idxs = []  # image indices
    y_idxs = []

    def __init__(self, x, y, z, image_idxs):
        self.x, self.y, self.z = x, y, z
        self.x_digits = [int(d) for d in str(x)]
        self.y_digits = [int(d) for d in str(y)]
        self.x_idxs = [choice(image_idxs[i]) for i in self.x_digits]
        self.y_idxs = [choice(image_idxs[i]) for i in self.y_digits]

    def __str__(self):
        return "{:d}, {:d}, {}\n{}\n{}".format(self.x, self.y, self.z,
                                               self.x_idxs, self.y_idxs)


class Example_Monadic:
    x = None  # input
    y = None  # output
    x_idxs = []  # image indices

    def __init__(self, x, y, image_idxs):
        self.x, self.y = x, y
        self.x_idxs = [choice(image_idxs[i]) for i in self.x]

    def __str__(self):
        return "{}, {:d}\n{}".format(self.x, self.y, self.x_idxs)


class Example_Sort:
    x = None  # input
    y = None  # output
    srtd = False  # if it is sorted
    x_idxs = []  # image indices

    def __init__(self, x, y, srtd, image_idxs):
        self.x, self.y, self.srtd = x, y, srtd
        self.x_idxs = [choice(image_idxs[i]) for i in self.x]

    def __str__(self):
        return "{}, {}, {}\n{}".format(self.x, self.y, self.srtd, self.x_idxs)


class Dataset_Dyadic(Dataset):

    def __init__(self, pairs, targets, imgs_data):
        self.imgs_data = imgs_data
        self.pairs = pairs
        self.targets = torch.LongTensor(targets)

    def __getitem__(self, index):
        i = self.pairs[index][0]
        j = self.pairs[index][1]
        img1 = self.imgs_data[i][0]
        img2 = self.imgs_data[j][0]
        # img1 = img1.view(img1.shape[0], -1)
        # img2 = img2.view(img2.shape[0], -1)
        img = torch.stack([img1, img2])
        target = int(self.targets[index])
        return img, target

    def __len__(self):
        return len(self.pairs)

    def get_data(self, use_cuda=True):
        p0 = self.pairs[0]
        img1 = self.imgs_data[p0[0]][0]
        img2 = self.imgs_data[p0[1]][0]
        img_tensor = torch.stack([img1, img2])
        img_tensor = torch.reshape(img_tensor, (1, 2, 1, 28, 28))
        i = 1
        while i < len(self.pairs):
            p0 = self.pairs[i]
            img1 = self.imgs_data[p0[0]][0]
            img2 = self.imgs_data[p0[1]][0]
            img = torch.stack([img1, img2])
            img = torch.reshape(img, (1, 2, 1, 28, 28))
            img_tensor = torch.cat((img_tensor, img), 0)
            i = i + 1
        if use_cuda:
            img_tensor = img_tensor.to(torch.device("cuda"))
        return img_tensor


def load_mnist():
    """
    Load MNIST dataset
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)

    return dataset1, dataset2


def group_mnist(dataset):
    """
    Group the dataset with the class labels
    """

    target_idxs = []
    for i in range(0, 10):
        idx = dataset.targets == i
        idx1 = [j for (j, v) in enumerate(idx) if v == True]
        target_idxs.append(idx1)
    return target_idxs


def get_mnist_imgs(dataset, indices, use_cuda=True):
    """
    Given get the image tensor from mnist dataset by indices
    """

    n = len(indices)
    img_tensor, tgt = dataset[indices[0]]
    img_tensor = torch.reshape(img_tensor, (1, 1, 28, 28))
    targets = [tgt]
    i = 1
    while i < n:
        img, tgt = dataset[indices[i]]
        img = torch.reshape(img, (1, 1, 28, 28))
        img_tensor = torch.cat((img_tensor, img), 0)
        targets.append(tgt)
        i = i + 1
    if use_cuda:
        img_tensor = img_tensor.to(torch.device("cuda"))
    return img_tensor, targets


def gen_dataset(Num, fun, img_idxs, min_val=0, max_val=100):
    """
    Generate dataset
    """

    i = 0
    re = []
    while i < Num:
        x = randint(min_val, max_val)
        y = randint(min_val, max_val)
        try:
            z = fun(x, y)
            i = i + 1
            ex = Example(x, y, z, img_idxs)
            re.append(ex)
        except:
            continue
    return re


def gen_dataset_monadic(Num, fun, img_idxs, min_val=0, max_val=9, min_len=2, max_len=10):
    """
    Generate monadic dataset, x is a list of digits (length in range from min_len to max_len)
    """
    i = 0
    re = []
    while i < Num:
        L = randint(min_len, max_len)
        x = [randint(min_val, max_val) for j in range(0, L)]
        try:
            y = fun(x)
            i = i + 1
            ex = Example_Monadic(x, y, img_idxs)
            re.append(ex)
        except:
            continue
    return re


def gen_dataset_sort(Num, img_idxs, min_val=0, max_val=9, length=5, reverse=True):
    """
    Generate sort(x,y) dataset, where x is the input list,
    y is the output permutation
    """
    i = 0
    re = []
    rg = list(range(min_val, max_val+1))
    while i < Num:
        x = sample(rg, length)
        y = sort_permutation(x, reverse=reverse)
        # z is 1 if x is already sorted
        z = 1 if all(y[i] <= y[i+1]
                     for i in range(len(y)-1)) else 0
        ex = Example_Sort(x, y, z, img_idxs)

        re.append(ex)
        i = i + 1
    return re


'''
Tasks
'''


def mysum(x):
    "monadic, sum of list of ints"
    return int(np.sum(x))


def myprod(x):
    "monadic, product of list of ints"
    return int(np.prod(x))


def myprod_sum(x):
    "monadic, ((a*b)+c)*d..."
    i = 1
    re = x[0]
    while i < len(x):
        if i % 2 == 1:
            re = re * x[i]
        else:
            re = re + x[i]
        i = i + 1
    return re


def ex1(x, y):
    return x + y


def ex2(x, y):
    return x - y


def ex3(x, y):
    return x * y


def ex4(x, y):
    assert y != 0
    return x / y


def ex5(x, y):
    return x * x + y * y


def ex6(x, y):
    return x * (x + y)


def ex7(x, y):
    return y * (x - y) * x


def ex8(x, y):
    return (x + y) * (x - y)


def ex9(x, y):
    return (x + x) * (y + y)


def ex10(x, y):
    assert y != 0
    return (x / y) * ((x - y) / y)


def sort_permutation(L: list, reverse=True) -> list:
    L1 = L.copy()
    L1.sort(reverse=reverse)
    p = [L1.index(e) + 1 for e in L]
    return p
