import asyncio
from perception import pseudo_label_probs


# Perceives the sample of images in each example and convert the result to a
# Prolog background knowledge file
def perception_to_kb(model, label_names, sample_indices,
                     all_examples, all_imgs_data,
                     clp_range='0..9',
                     bk_file='arithmetic_bk.pl',
                     path='../prolog/tmp_bk.pl',
                     use_cuda=True):
    header = ":- ['{}'].\n\nuser:clp_range('{}').\n\n".format(
        bk_file, clp_range)
    # use perception model to calculate the probabilistic distribution p(z|x)
    prob_dist = pseudo_label_probs(model, sample_indices,
                                   all_examples, all_imgs_data, use_cuda=use_cuda)
    # 1a. generate variable list
    # 1b. generate string of metagol example "[f(['X1','X2'],y1),f(['X3','X4'],y2)...]"
    var_list, pos_sample_str = gen_vnames_and_exlist(
        sample_indices, all_examples)
    # 2. generate string of all "nn('X1',1,p1).\nnn('X1',2,p2)..."
    prob_facts = gen_prob_facts(var_list, label_names, prob_dist)
    # generate string of query "learn :- Pos=..., metaabd(Pos,[])."
    query_str = "\na :- Pos={}, metaabd(Pos).\n".format(pos_sample_str)
    with open(path, 'w') as pl:
        pl.write(header)
        pl.write(prob_facts)
        pl.write(query_str)


# Examples and variable names
def gen_vnames_and_exlist(sample_indices, all_examples):
    # variable names[X1,X2] and examples string [f([X1,X2,..],y),...]
    re1 = []
    re2 = "["
    cnt = 0
    for i in sample_indices:
        sample = all_examples[i]
        vlist = []
        in_str = "["
        for j in range(len(sample.x)):
            vname = 'X' + str(cnt)
            vlist.append(vname)
            in_str = in_str + "'" + vname + "',"
            cnt = cnt + 1
        in_str = in_str[:-1] + "]"
        ex_str = "f({0:s},{1:d})".format(in_str, sample.y)
        re1.append(vlist)
        re2 = re2 + ex_str + ","
    re2 = re2[:-1] + "]"
    return re1, re2


# Generate probabilistic facts
def gen_prob_facts(var_list, label_names, prob_dist):
    n = len(var_list)
    assert len(prob_dist) == n
    re = ""
    for i in range(n):
        # for each example
        m = len(var_list[i])
        assert len(prob_dist[i]) == m
        for j in range(m):
            # for each 'X'
            vname = var_list[i][j]
            prob = prob_dist[i][j]
            for k in range(len(label_names)):
                lname = label_names[k]
                fact_str = "nn('{}',{},{}).\n".format(vname, lname, prob[k])
                re = re + fact_str
    return re


# Run prolog to get the output.
# Return the STDOUT and error codes (-1 for runtime error, -2 for timeout)
async def run_pl(file_path='../prolog/tmp_bk.pl', timeout=10):
    # cmd = "swipl --stack-limit=8g -s {} -g a -t halt; rm -f {}".format(file_path, file_path)
    # DEBUG: no rm
    cmd = "swipl --stack-limit=8g -s {} -g a -t halt".format(file_path)
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)

    try:
        # 2 seconds timeout
        stdout, stderr = await asyncio.wait_for(proc.communicate(),
                                                timeout=timeout)
        if proc.returncode == 0:
            return 0, stdout.decode('UTF-8')
        else:
            return -1, stderr.decode('UTF-8')  # runtime error
    except asyncio.TimeoutError as e:
        if proc.returncode is None:
            proc.kill()
        return -2, "Timeout " + str(e)  # timeout error


# Get the output results, which are abduced labels and the hypothesis.
def parse_pl_result(pl_out_str, sample_indices, label_names, all_examples):
    prog_str, labels_str = read_pl_out(pl_out_str)
    labels_flatten = [label_names.index(
        int(s)) for s in labels_str.split(',')]
    imgs_flatten = []

    for idx in sample_indices:
        sample = all_examples[idx]
        # print(sample.x, sample.x_idxs)
        imgs_flatten = imgs_flatten + sample.x_idxs

    return prog_str, labels_flatten, imgs_flatten


def read_pl_out(pl_out_str):
    prog = ""
    labels = None

    prog_start = False
    label_start = False
    for line in pl_out_str.splitlines():
        if line[0] == '-':
            if line[2:-1] == 'Program':
                prog_start = True
                continue
            elif line[2:-1] == 'Abduced Facts':
                prog_start = False
                label_start = True
                continue
        if prog_start:
            prog = prog + line + "\n"
        if label_start:
            labels = line
    return prog, labels[1:-1]
