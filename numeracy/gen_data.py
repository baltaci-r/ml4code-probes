import numpy as np, random, num2words, pandas as pd, argparse
from nltk import CFG
from nltk.parse.generate import generate

DATA_TYPES = ['floats', 'ints', 'words', 'negatives']
TASKS = ['list_maximum', 'decoding', 'addition', 'logic']
LOGIC_TYPES = ['bool_simple', 'bool', 'algebraic']
TYPES = DATA_TYPES + LOGIC_TYPES

# CFG: (python)  # todo: do the same for other languages!
BOOL_CFG_SIMPLE = """
bexp -> "True" | "False" | bexp "or" bexp | bexp "and" bexp | "not" bexp
"""

BOOL_CFG = """
bexp -> "True" | "False" | bexp "or" bexp | bexp "and" bexp | "not" bexp | "(" bexp ")"
"""

ARITH_CFG = """
expr -> "INT" | "VAR" | "(" expr ")" | "+" expr | "-" expr | expr "/" expr |  expr "*" expr | expr "+" expr | expr "-" expr
"""

def gen_list_maximum(data_type, min, max, n: int, m: int):
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


def gen_decoding(dtype, min, max, n: int, m: int):
    assert dtype in DATA_TYPES
    assert m==1
    if dtype == 'ints':
        outs = pd.DataFrame(np.random.randint(low=int(min), high=int(max), size=(n,)))
        inps = pd.DataFrame(outs, dtype=str)
    elif dtype == 'float':
        outs = pd.DataFrame(np.random.uniform(low=min, high=max, size=(n,)))
        inps = pd.DataFrame(outs, dtype=str)
    elif dtype == 'words':
        assert max < 100
        outs = pd.DataFrame(np.random.randint(low=int(min), high=int(max), size=(n,)))
        inps = outs.applymap(lambda x: num2words.num2words(x))


def gen_addition(dtype, min, max, n: int, m: int):
    assert dtype in DATA_TYPES
    if dtype == 'ints':
        inps = pd.DataFrame(np.random.randint(low=int(min), high=int(max), size=(n, m)))
        outs = inps.sum(axis=0)
    elif dtype == 'float':
        inps = pd.DataFrame(np.random.uniform(low=min, high=max, size=(n, m)))
        outs = inps.sum(axis=0)
    elif dtype == 'words':
        assert max < 100
        rand_ints = pd.DataFrame(np.random.randint(low=int(min), high=int(max), size=(n, m)))
        inps = rand_ints.applymap(lambda x: num2words.num2words(x))
        outs = rand_ints.sum(axis=0)


def gen_logic_operations(dtype, min, max, n: int, m: int):
    """
    booleans depend on the language and how the model is trained (e.g. tag)
    simple example: True and not False or True and not True and not False : True
    adding hierarchy: ((True and not False) or (True and True)) and False : False
    adding variables: a=10; b=5; a>5 and b>2 : True
    more than one output: a=10; b=5; a>5 and b>2: True, a<5 and b>2 : False, a>=10 and b>2 : True
    mathematical expressions: Regression task
    """

    # todo include min max to truncate operations in between lengths
    assert dtype in LOGIC_TYPES
    if dtype == 'bool_simple':
        grammar = CFG.fromstring(BOOL_CFG_SIMPLE)
        inps = random.sample(list(generate(grammar, depth=m)), n)
        inps = pd.DataFrame([' '.join(sent) for sent in inps])
        inps = inps.astype(str)
        pythonizer = lambda x: eval(f"{x}")
        outs = inps.applymap(pythonizer)

    elif dtype == 'bool':
        grammar = CFG.fromstring(BOOL_CFG)
        inps = random.sample(list(generate(grammar, depth=m)), n)
        inps = pd.DataFrame([' '.join(sent) for sent in inps])
        inps = inps.astype(str)
        pythonizer = lambda x: eval(f"{x}")
        outs = inps.applymap(pythonizer)

    elif dtype == 'algebraic':
        grammar = CFG.fromstring(ARITH_CFG)
        inps = random.sample(list(generate(grammar, depth=m)), n)
        inps = pd.DataFrame([' '.join(sent) for sent in inps])
        inps = inps.astype(str)
        pythonizer = lambda x: eval(f"{x}")
        outs = inps.applymap(pythonizer)
        A=1


def run_task(args):
    run = {
        'list_maximum': gen_list_maximum,  # (args.type, args.min, args.max, args.n, args.m),
        'decoding': gen_decoding,  # (args.type, args.min, args.max, args.n, 1),
        'addition': gen_addition,  # (args.type, args.min, args.max, args.n, args.m),
        'logic': gen_logic_operations  # (args.type, args.min?, args.max?, args.n, args.m)
    }
    return run[args.task](args.type, args.min, args.max, args.n, args.m)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--min', type=float)
    parser.add_argument('--max', type=float)
    parser.add_argument('-n', type=int)
    parser.add_argument('-m', type=int)
    parser.add_argument('--type', choices=TYPES)
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('-s', type=float, help='train-test split')

    args = parser.parse_args()
    run_task(args)
