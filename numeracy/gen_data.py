import numpy as np, random, num2words, pandas as pd, argparse

DATA_TYPES = ['floats', 'ints', 'words', 'negatives']
TASKS = ['list_maximum', 'decoding', 'addition']


def gen_list_maximum(min, max, n, m, data_type):
    if data_type == 'ints':
        inps = pd.DataFrame(np.random.randint(low=int(min), high=int(max), size=(n, m)))
        outs = pd.DataFrame((inps == np.max(inps, axis=0)), dtype=int)
    elif data_type == 'float':
        inps = pd.DataFrame(np.random.uniform(low=min, high=max, size=(n, m)))
        outs = pd.DataFrame((inps == np.max(inps, axis=0)), dtype=int)
    elif data_type == 'words':
        assert max < 100
        rand_ints = pd.DataFrame(np.random.randint(low=int(min), high=int(max), size=(n, m)))
        inps = rand_ints.applymap(lambda x: num2words.num2words(x))
        outs = pd.DataFrame((rand_ints == np.max(rand_ints, axis=0)), dtype=int)


def gen_decoding(min, max, n, data_type):
    if data_type == 'ints':
        outs = pd.DataFrame(np.random.randint(low=int(min), high=int(max), size=(n,)))
        inps = pd.DataFrame(outs, dtype=str)
    elif data_type == 'float':
        outs = pd.DataFrame(np.random.uniform(low=min, high=max, size=(n,)))
        inps = pd.DataFrame(outs, dtype=str)
    elif data_type == 'words':
        assert max < 100
        outs = pd.DataFrame(np.random.randint(low=int(min), high=int(max), size=(n,)))
        inps = outs.applymap(lambda x: num2words.num2words(x))


def gen_addition(min, max, n, m, data_type):
    if data_type == 'ints':
        inps = pd.DataFrame(np.random.randint(low=int(min), high=int(max), size=(n, m)))
        outs = inps.sum(axis=0)
    elif data_type == 'float':
        inps = pd.DataFrame(np.random.uniform(low=min, high=max, size=(n, m)))
        outs = inps.sum(axis=0)
    elif data_type == 'words':
        assert max < 100
        rand_ints = pd.DataFrame(np.random.randint(low=int(min), high=int(max), size=(n, m)))
        inps = rand_ints.applymap(lambda x: num2words.num2words(x))
        outs = rand_ints.sum(axis=0)


def gen_logic_operations():
    # booleans depend on the language and how the model is trained (e.g. tag)
    # simple example: True and False or True and True and False : False
    # adding hierarchy: ((True and False) or (True and True)) and False : False
    # adding variables: a=10; b=5; a>5 and b>2 : True
    # maybe more than one output: a=10; b=5; a>5 and b>2: True, a<5 and b>2 : False, a>=10 and b>2 : True
    A=1



def run_task(args):
    run = {
        'list_maximum': gen_list_maximum(args.min, args.max, args.n, args.m, args.dt),
        'decoding': gen_decoding(args.min, args.max, args.n, args.dt),
        'addition': gen_addition(args.min, args.max, args.n, args.m, args.dt),

    }
    return run[args.task]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--min', type=float)
    parser.add_argument('--max', type=float)
    parser.add_argument('-n', type=int)
    parser.add_argument('-m', type=int)
    parser.add_argument('-dt', choices=DATA_TYPES)
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('-s', type=float, help='train-test split')

    args = parser.parse_args()
    run_task(args)
