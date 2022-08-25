import asyncio
import random
import itertools
import operator
import torch
from abduce import run_pl, parse_pl_result, perception_to_kb
from perception import train, test


# Sample batches
def sample_batches(all_examples, batch_size, shuffle=True):
    n_samples = len(all_examples)
    n_batches = n_samples // batch_size + \
        (1 if n_samples % batch_size > 0 else 0)
    re = []

    indices = list(range(n_samples))
    if shuffle:
        random.shuffle(indices)
    for i in range(n_batches):
        batch = indices[i*batch_size:(i+1)*batch_size]
        re.append(batch)
    return re


# Sample curriculum
def sample_curriculums(all_examples, batch_vars, n_batches=500, shuffle=True):
    print("Sampling batches with {} variables".format(batch_vars), end="...")
    n_samples = len(all_examples)
    re = []

    indices = list(range(n_samples))
    if shuffle:
        random.shuffle(indices)

    skip_list = []
    i = 0
    while i < n_samples and len(re) < n_batches:
        if i % 1000 == 0:
            print(i, end="...")
        idx = indices[i]
        i = i + 1
        if idx in skip_list:
            continue
        else:
            skip_list.append(idx)
            n_vars = len(all_examples[idx].x_idxs)
            if batch_vars >= n_vars and batch_vars - n_vars <= 1:
                batch = [idx]
                re.append(batch)
            elif batch_vars - n_vars >= 2:
                batch = [idx]
                for j in range(n_samples):
                    if n_vars >= batch_vars:
                        break
                    idx2 = indices[j]
                    if idx2 in skip_list:
                        continue
                    else:
                        j_vars = len(all_examples[idx2].x_idxs)
                        if n_vars+j_vars <= batch_vars:
                            batch.append(idx2)
                            skip_list.append(idx2)
                            n_vars = n_vars+j_vars
                        else:
                            break
                re.append(batch)
    print("done ({} batches).".format(len(re)))
    return re

# Learn neural net with metagol induction


async def train_abduce(batches, model, optimizer, scheduler,
                       label_names, all_examples,
                       train_img_data, test_img_data, train_img_labels,
                       pl_file_path='../prolog/tmp_bk.pl',
                       clp_range='0..9',
                       timeout=10, device=torch.device("cuda"), epochs=10,
                       log_interval=500, **kwargs):
    train_img_indices = []  # images for updating neural model
    train_img_targets = []  # logically abduced labels

    print("Start abductive metagol:")
    for i, samples in enumerate(batches):
        # get the probabilistic facts
        perception_to_kb(model, label_names, samples,
                         all_examples, train_img_data,
                         path=pl_file_path, clp_range=clp_range)
        # call metagol to abduce the logical hypothesis and image labels
        pl_err, pl_out = await run_pl(file_path=pl_file_path, timeout=timeout)
        if pl_err == 0:
            # gather the outputs
            prog_str, labels_flatten, img_indices_flatten = parse_pl_result(
                pl_out, samples, label_names, all_examples)

            train_img_targets = train_img_targets + labels_flatten
            train_img_indices = train_img_indices + img_indices_flatten

            if i % 100 == 0:
                print("Abduction Turn {}:".format(i))
                print(prog_str)
                print([label_names[l] for l in labels_flatten])
                for s in samples:
                    print(all_examples[s].x, end=' ')
                print()
            else:
                print(i, end=" ")
        else:
            # if anything wrong happend, just skip this batch
            if i % 100 == 0:
                print("Abduction Turn {}:".format(i) + pl_out)
            else:
                print("[{}]".format(i), end=" ")
            continue

    # Change the targets of training images
    acc = 0
    for i, img in enumerate(train_img_indices):
        train_img_data.targets[img] = train_img_targets[i]
        if train_img_targets[i] == train_img_labels[img]:
            acc = acc + 1
    print("\nAbduced Label Acc: {}".format(acc/len(train_img_indices)))

    sup_imgs_train = torch.utils.data.Subset(train_img_data, train_img_indices) # only use the successfuly abduced labels
    sup_train_loader = torch.utils.data.DataLoader(sup_imgs_train, **kwargs)
    sup_test_loader = torch.utils.data.DataLoader(test_img_data, **kwargs)

    # Update neural model
    for epoch in range(1, epochs):
        train(model, device, sup_train_loader,
              optimizer, epoch, log_interval)
        test(model, device, sup_test_loader)
        scheduler.step()
    return None


async def abduce_coroutine(i, model, label_names, samples, all_examples,
                           train_img_data, bk_file, pl_file_dir, clp_range,
                           timeout, use_cuda=True):
    pl_file_path = pl_file_dir + '{}_bk.pl'.format(i)
    perception_to_kb(model, label_names, samples,
                     all_examples, train_img_data,
                     bk_file=bk_file,
                     path=pl_file_path,
                     clp_range=clp_range,
                     use_cuda=use_cuda)

    pl_err, pl_out = await run_pl(file_path=pl_file_path, timeout=timeout)
    if pl_err != 0:
        print("[{}]".format(i), end=" ")
    else:
        print("{}!".format(i), end=" ")
    return {'err': pl_err, 'out': pl_out}


async def safe_abduce(i, sem, model, label_names, samples, all_examples,
                      train_img_data, bk_file, pl_file_dir, clp_range, timeout, use_cuda=True):
    async with sem:  # semaphore limits num of simultaneous downloads
        return await abduce_coroutine(i, model, label_names, samples,
                                      all_examples, train_img_data,
                                      bk_file,
                                      pl_file_dir, clp_range, timeout, use_cuda=use_cuda)


async def abduce_concurrent(sem, batches, model, label_names,
                            all_examples, train_img_data,
                            bk_file,
                            pl_file_dir, clp_range, timeout, use_cuda=True):
    tasks = [
        # creating task starts coroutine
        asyncio.ensure_future(safe_abduce(i, sem, model, label_names, samples,
                                          all_examples, train_img_data,
                                          bk_file,
                                          pl_file_dir, clp_range, timeout, use_cuda=use_cuda))
        for i, samples
        in enumerate(batches)
    ]
    return await asyncio.gather(*tasks)  # await moment all tasks done


# Learn neural net with metagol induction
def train_abduce_concurrent(loop, sem, batches, model, optimizer, scheduler,
                            label_names, all_examples,
                            train_img_data, test_img_data,
                            train_img_labels,
                            bk_file='../arithmetic_bk.pl',
                            pl_file_dir='../prolog/tmp/',
                            clp_range='0..9',
                            timeout=10, device=torch.device("cuda"),
                            epochs=10, log_interval=500, **kwargs):
    train_img_indices = []  # images for updating neural model
    train_img_targets = []  # logically abduced labels
    if device == torch.device("cuda"):
        use_cuda = True
    else:
        use_cuda = False

    # Start concurrent abduction
    try:
        tasks_results = loop.run_until_complete(
            abduce_concurrent(sem, batches, model, label_names,
                              all_examples, train_img_data,
                              bk_file,
                              pl_file_dir, clp_range, timeout,
                              use_cuda=use_cuda)
        )
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())

    succ = 0
    progs = []
    for i, samples in enumerate(batches):
        pl_err = tasks_results[i]['err']
        pl_out = tasks_results[i]['out']
        if pl_err == 0:
            succ = succ + 1
            # gather the outputs
            prog_str, labels_flatten, img_indices_flatten = parse_pl_result(
                pl_out, samples, label_names, all_examples)
            # print("\n{}:".format(i))
            # print([label_names[l] for l in labels_flatten])
            # for s in samples:
            #     print(all_examples[s].x, end=' ')
            train_img_targets = train_img_targets + labels_flatten
            train_img_indices = train_img_indices + img_indices_flatten
            progs.append(prog_str)
        else:
            # if anything wrong happend, just skip this batch
            continue

    print("\nMost Frequent Program:")
    print(most_common(progs), end="")
    print("Successfully abduced batches: {}/{}".format(succ, len(batches)))

    # Change the targets of training images
    acc = 0
    for i, img in enumerate(train_img_indices):
        train_img_data.targets[img] = train_img_targets[i]
        if train_img_targets[i] == train_img_labels[img]:
            acc = acc + 1

    print("Abduced label Acc: {}".format(acc/len(train_img_indices)))

    sup_imgs_train = torch.utils.data.Subset(train_img_data, train_img_indices)
    sup_train_loader = torch.utils.data.DataLoader(sup_imgs_train, **kwargs)
    sup_test_loader = torch.utils.data.DataLoader(test_img_data, **kwargs)

    # Update neural model
    for epoch in range(0, epochs):
        train(model, device, sup_train_loader,
              optimizer, epoch, log_interval)
        test(model, device, sup_test_loader)
        scheduler.step()
    return tasks_results


def most_common(L):
    # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(L))
    # print 'SL:', SL
    groups = itertools.groupby(SL, key=operator.itemgetter(0))
    # auxiliary function to get "quality" for an item

    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index
    # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]
