import asyncio
import random

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import yaml

from learn import sample_curriculums, train_abduce_concurrent
from data import load_mnist, group_mnist
from perception import Net, train

def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


def one_shot_pretrain(model, mnist_imgs_train,
                      device=torch.device('cuda'), **kwargs):
    n_samples = 1
    EPOCHS = 20
    LR = 1
    GAMMA = 0.7
    LOG_INTERVAL = 500

    print("Sample {} example in each class".format(n_samples))

    groups = group_mnist(mnist_imgs_train)
    few_shot_indices = []

    for i in range(10):
        few_shot_indices = few_shot_indices + \
            random.sample(groups[i], n_samples)

    # few_shot_indices = random.sample(all_img_indices, n_samples)

    sup_imgs_train = torch.utils.data.Subset(
        mnist_imgs_train, few_shot_indices)

    sup_train_loader = torch.utils.data.DataLoader(
        sup_imgs_train, **kwargs)

    optimizer = optim.Adadelta(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=1, gamma=GAMMA)

    for epoch in range(1, EPOCHS):
        train(model, device, sup_train_loader,
              optimizer, epoch, LOG_INTERVAL)
        scheduler.step()


def main():
    task_name = 'mysum_full'
    data_path = '../data/monadic/'
    tmp_bk_dir = '../prolog/tmp_pl/'
    label_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    clp_range = '0..9'
    n_labels = 10  # number of labels

    ONE_SHOT_PRETRAIN = False
    # For reproductivity
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)

    file_name = data_path + task_name + '.yaml'
    with open(file_name, 'r') as f:
        examples = yaml.load(f, Loader=yaml.Loader)

    examples = examples[0:3000]

    imgs_train, imgs_test = load_mnist()

    imgs_train_targets = imgs_train.targets.tolist()

    all_img_indices = []
    for e in examples:
        all_img_indices = all_img_indices + e.x_idxs

    device = torch.device("cuda")
    p_model = Net(n_labels).to(device)
    print(p_model)
    p_model.apply(weights_init)

    # Learning with abduction
    EPOCHS = 100
    N_BATCHES = 3000
    N_CORES = 32  # number of parallel abductions
    NN_EPOCHS = 1
    LR = 0.01
    r = 0.005 # speed for adapting learning rate
    GAMMA = 1.0
    LOG_INTERVAL = 500
    NUM_WORKERS = 32
    BATCH_VARS = 7 # maximum number of variables in abduction

    kwargs = {'batch_size': 64}
    kwargs.update({'num_workers': NUM_WORKERS,
                   'pin_memory': True,
                   'shuffle': True},
                  )

    if ONE_SHOT_PRETRAIN:
        one_shot_pretrain(p_model, imgs_train, **kwargs)

    sem = asyncio.Semaphore(N_CORES)
    loop = asyncio.get_event_loop()

    try:
        for T in range(EPOCHS):
            lr_t = LR*(1 + r)**T # increase learning rate during iteration

            optimizer = optim.SGD(p_model.parameters(), lr=lr_t)
            scheduler = StepLR(optimizer, step_size=1, gamma=GAMMA)

            print("======\nEpoch {}\n======".format(T))
            # new batch assignment to break the dependency
            batches = sample_curriculums(
                examples, BATCH_VARS, n_batches=N_BATCHES, shuffle=True)
            train_abduce_concurrent(loop, sem, batches, p_model,
                                    optimizer, scheduler,
                                    label_names, examples,
                                    imgs_train, imgs_test,
                                    imgs_train_targets,
                                    bk_file='../arithmetic_bk.pl',
                                    pl_file_dir=tmp_bk_dir,
                                    clp_range=clp_range,
                                    epochs=NN_EPOCHS, timeout=100,
                                    log_interval=LOG_INTERVAL,
                                    device=device,
                                    **kwargs)
            torch.save(p_model, 'models/sum_full_null.pt')
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


if __name__ == "__main__":
    main()
