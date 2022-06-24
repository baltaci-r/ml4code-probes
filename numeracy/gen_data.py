import numpy as np, random, num2words, pandas as pd, argparse, re
from nltk import CFG
from nltk.parse.generate import generate

DATA_TYPES = ['floats', 'ints', 'words', 'negatives']
TASKS = ['list_maximum', 'decoding', 'addition', 'logic']
LOGIC_TYPES = ['bool_simple', 'bool', 'algebraic_int', 'algebraic_float', \
               'var+algebraic_int', 'var+algebraic_float', 'var+comparative_int', 'var+comparative_float']
TYPES = DATA_TYPES + LOGIC_TYPES

# CFG: (python)  # todo: do the same for other languages!
BOOL_CFG_SIMPLE = """
bexp -> "True" | "False" | bexp "or" bexp | bexp "and" bexp | "not" bexp
"""
BOOL_CFG = """
bexp -> "True" | "False" | bexp "or" bexp | bexp "and" bexp | "not" bexp | "(" bexp ")"
"""

ARITH_CFG_INT = """
expr -> "NUM" | "(" expr ")" | "+" expr | "-" expr | expr "+" expr | expr "-" expr
"""
ARITH_CFG = """
expr -> "NUM" | "(" expr ")" | "+" expr | "-" expr | expr "/" expr |  expr "*" expr | expr "+" expr | expr "-" expr
"""
ARITH_VAR_CFG_INT = """
expr -> "NUM" | "VAR" | "(" expr ")" | "+" expr | "-" expr | expr "+" expr | expr "-" expr
"""
ARITH_VAR_CFG = """
expr -> "NUM" | "VAR" | "(" expr ")" | "+" expr | "-" expr | expr "/" expr |  expr "*" expr | expr "+" expr | expr "-" expr
"""

OPS = ['<', '>', '>=', '<=', '==']


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

    def set_value(x):
        while re.search('NUM', x):
            i, j = re.search('NUM', x).regs[0]
            if 'int' in dtype:
                x = x[:i] + str(random.randint(min, max)) + x[j:]
            elif 'float' in dtype:
                x[i:j] = x[:i] + str(random.uniform(min, max)) + x[j:]
        return x

    def set_value_var(x):
        x = set_value(x)
        if 'int' in dtype:
            x = f'VAR = {random.randint(min, max)}; ' + x
        elif 'float' in dtype:
            x = f'VAR = {random.uniform(min, max)}; ' + x
        return x

    def eval_var(exp):
        loc = {}
        exp = exp.split(';')
        exp = '\nout ='.join(exp)
        exec(exp, globals(), loc)
        return loc['out']

    def eval_comp(exp, nvar=2):
        loc = {}
        exp = exp.split(';')
        exp = [e.strip() for e in exp]
        exp = '\n'.join(exp[:nvar]) + '\n' + '\n'.join([f'out_{i}= '+ e for i, e in enumerate(exp[nvar:])])
        exec(exp, globals(), loc)
        outs = [v for k,v in loc.items() if 'out' in k]
        return outs

    def gen_comp(m):
        """
        simple: X = 1; Y = 2; X > Y; X==Y; X < Y (m comparisons)
        X (<|>|>=|<=|==) Y
        complex: X = 1; Y = 2; X > 3 and Y <1; X < 2 or Y >= 5; X == 0 and Y > 3; (m comparisons)
        X (<|>|>=|<=|==) int/float; (and|or) Y (<|>|>=|<=|==) int/float;
        """
        if 'int' in dtype:
            exp = f'X = {random.randint(min, max)}; Y = {random.randint(min, max)}; '
        elif 'float' in dtype:
            exp = f'X = {random.uniform(min, max)}; Y = {random.uniform(min, max)}; '

        for i in range(m):
            if i==m-1:
                exp += f'X {random.choice(OPS)} Y'
            else:
                exp += f'X {random.choice(OPS)} Y; '
        return exp




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
    # todo: attach several of these adn generate multiple output
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

    elif dtype == 'var_bool':
        grammar = CFG.fromstring(BOOL_CFG)
        inps = random.sample(list(generate(grammar, depth=m)), n)
        inps = pd.DataFrame([' '.join(sent) for sent in inps])
        inps = inps.astype(str)
        pythonizer = lambda x: eval(f"{x}")  # todo: check if we need formatting
        outs = inps.applymap(pythonizer)

    elif dtype in ['algebraic_int', 'algebraic_float']:
        cfg = ARITH_CFG_INT if dtype == 'algebraic_int' else ARITH_CFG  # if floats > wide range of outs!
        grammar = CFG.fromstring(cfg)
        inps = random.sample(list(generate(grammar, depth=m)), n)
        inps = pd.DataFrame([' '.join(sent) for sent in inps])
        inps = inps.applymap(set_value)
        pythonizer = lambda x: eval(x)
        outs = inps.applymap(pythonizer)

    elif dtype in ['var+algebraic_int', 'var+algebraic_float']:
        cfg = ARITH_VAR_CFG_INT if dtype == 'var+algebraic_int' else ARITH_VAR_CFG
        grammar = CFG.fromstring(cfg)
        inps = random.sample(list(generate(grammar, depth=m)), n)
        inps = pd.DataFrame([' '.join(sent) for sent in inps])
        inps = inps.applymap(set_value_var)
        eval_var_func = lambda x: eval_var(x)
        outs = inps.applymap(eval_var_func)

    elif dtype in ['var+comparative_int', 'var+comparative_float']:
        # defaults to two variables # todo: add more variables?
        inps = pd.DataFrame([gen_comp(m) for _ in range(n)])
        eval_var_func = lambda x: eval_comp(x)
        outs = inps.applymap(eval_var_func)
        outs = pd.DataFrame(outs[0].to_list())


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
