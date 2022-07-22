import asyncio
import ast

import torch

from perception import predict_idx_labels
from abduce import run_pl


async def eval_model_monadic(i, model, learned_prog, pl_file_dir, ex_id,
                             all_examples, imgs_data, label_names):
    ex = all_examples[ex_id]
    predicted = predict_idx_labels(
        model, imgs_data, ex.x_idxs, label_names)
    query_str = ":-['{}'].\n\na:-f({},Y),writeln(Y).".format(
        learned_prog, str(predicted))

    pl_file_path = pl_file_dir + '{}_bk.pl'.format(i)
    with open(pl_file_path, 'w') as pl:
        pl.write(query_str)

    pl_err, pl_out = await run_pl(file_path=pl_file_path, timeout=60)

    if pl_err != 0 and i % 1000 == 0:
        print("[{}]".format(i), end=" ")
    elif pl_err == 0 and i % 1000 == 0:
        print("{}!".format(i), end=" ")
    return {'err': pl_err, 'out': pl_out}


async def safe_eval_model_monadic(sem, i, model, learned_prog, pl_file_dir,
                                  ex_id, all_examples, imgs_data, label_names):
    async with sem:  # semaphore limits num of simultaneous downloads
        return await eval_model_monadic(i, model, learned_prog, pl_file_dir,
                                        ex_id, all_examples, imgs_data,
                                        label_names)


async def eval_model_monadic_concurrent(sem, model, label_names,
                                        all_examples, imgs_data,
                                        learned_prog, pl_file_dir):
    tasks = [
        # creating task starts coroutine
        asyncio.ensure_future(
            safe_eval_model_monadic(sem, i, model, learned_prog,
                                    pl_file_dir, i, all_examples,
                                    imgs_data, label_names))
        for i
        in range(len(all_examples))
    ]
    return await asyncio.gather(*tasks)  # await moment all tasks done


def evaluate_model_monadic(loop, sem, model, all_examples, imgs_data,
                           label_names, learned_prog, pl_file_dir, log=False):
    try:
        tasks_results = loop.run_until_complete(
            eval_model_monadic_concurrent(sem, model, label_names,
                                          all_examples, imgs_data,
                                          learned_prog, pl_file_dir)
        )
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())

    # collect all results
    targets = []
    pred = []
    for i in range(len(all_examples)):
        pl_err = tasks_results[i]['err']
        pl_out = tasks_results[i]['out']
        ex = all_examples[i]
        if pl_err == 0:
            targets.append(ex.y)
            pred.append(ast.literal_eval(pl_out))
        else:
            pred.append(0)

    print()

    if not log:
        test_loss = torch.nn.L1Loss(reduction='sum')(
            torch.FloatTensor(pred), torch.FloatTensor(targets)).item()
    else:
        test_loss = torch.nn.L1Loss(reduction='sum')(
            torch.log(torch.FloatTensor(pred)+1e-10),
            torch.log(torch.FloatTensor(targets)+1e-10)).item()
    test_loss /= len(all_examples)
    return test_loss
