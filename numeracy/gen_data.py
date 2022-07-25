import numpy as np, random, num2words, pandas as pd, argparse, re, json
from nltk import CFG
from nltk.parse.generate import generate

DATA_TYPES = ['floats', 'ints', 'words', 'negatives']
TASKS = ['list_maximum', 'decoding', 'addition', 'logic']
LOGIC_TYPES = ['bool_simple', 'bool', 'var_bool', 'algebraic_int', 'algebraic_float', \
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


def gen_list_maximum(dtype, min, max, n: int, m: int):
    assert dtype in DATA_TYPES
    if dtype == 'ints':
        # inps = pd.DataFrame(np.random.randint(low=int(min), high=int(max), size=(n, m))) #.astype(str)
        inps = np.random.randint(low=int(min), high=int(max), size=(n, m))  # .astype(str)
        outs = (inps == inps.max(axis=1)[:, None]).astype(int)
        inps = inps.astype(str)
    elif dtype == 'floats':
        inps = np.random.uniform(low=min, high=max, size=(n, m))
        outs = (inps == inps.max(axis=1)[:, None]).astype(int)
        inps = inps.astype(str)
    elif dtype == 'words':
        assert max < 100
        rand_ints = np.random.randint(low=int(min), high=int(max), size=(n, m))
        inps = pd.DataFrame(rand_ints).applymap(lambda x: num2words.num2words(x))
        outs = (rand_ints == rand_ints.max(axis=1)[:, None]).astype(int)
        inps = inps.values
    return inps.tolist(), outs.tolist()


def gen_decoding(dtype, min, max, n: int, m: int):
    assert dtype in DATA_TYPES
    assert m == 1
    if dtype == 'ints':
        outs = np.random.randint(low=int(min), high=int(max), size=(n, m))
        inps = outs.astype(str)
    elif dtype == 'floats':
        outs = np.random.uniform(low=min, high=max, size=(n, m))
        inps = outs.astype(str)
    elif dtype == 'words':
        assert max < 100
        outs = pd.DataFrame(np.random.randint(low=int(min), high=int(max), size=(n, m)))
        inps = outs.applymap(lambda x: num2words.num2words(x)).to_numpy()
        outs = outs.to_numpy()
    return inps.tolist(), outs.tolist()


def gen_addition(dtype, min, max, n: int, m: int):
    assert dtype in DATA_TYPES
    if dtype == 'ints':
        inps = np.random.randint(low=int(min), high=int(max), size=(n, m))
        outs = inps.sum(axis=1)
    elif dtype == 'floats':
        inps = np.random.uniform(low=min, high=max, size=(n, m))
        outs = inps.sum(axis=1)
    elif dtype == 'words':
        assert max < 100
        rand_ints = pd.DataFrame(np.random.randint(low=int(min), high=int(max), size=(n, m)))
        inps = rand_ints.applymap(lambda x: num2words.num2words(x)).values
        outs = rand_ints.sum(axis=1).to_numpy()
    return inps.astype(str).tolist(), outs.tolist()


def gen_logic_operations(dtype, min, max, n: int, m: int):

    # todo: for loop for other languages if any different
    def set_value(x):
        for i, y in enumerate(x):
            if y == 'NUM':
                if 'int' in dtype:
                    x[i] = str(random.randint(min, max))
                elif 'float' in dtype:
                    x[i] = str(random.uniform(min, max))
        return x

    def set_value_var(x):
        x = set_value(x)
        if 'int' in dtype:
            x = ['VAR', '=', str(random.randint(min, max)), ';'] + x
        elif 'float' in dtype:
            x = ['VAR', '=', str(random.uniform(min, max)), ';'] + x
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
        outs = [v for k, v in loc.items() if 'out' in k]
        return outs

    def gen_comp(m):
        """
        simple: X = 1; Y = 2; X > Y; X==Y; X < Y (m comparisons)
        X (<|>|>=|<=|==) Y
        complex: X = 1; Y = 2; X > 3 and Y <1; X < 2 or Y >= 5; X == 0 and Y > 3; (m comparisons)
        X (<|>|>=|<=|==) int/float; (and|or) Y (<|>|>=|<=|==) int/float;
        """
        if 'int' in dtype:
            exp = f'X = {random.randint(min, max)} ; Y = {random.randint(min, max)} ; '
        elif 'float' in dtype:
            exp = f'X = {random.uniform(min, max)} ; Y = {random.uniform(min, max)} ; '

        for i in range(m):
            if i==m-1:
                exp += f'X {random.choice(OPS)} Y'
            else:
                exp += f'X {random.choice(OPS)} Y ; '
        return exp.split()


    """
    booleans depend on the language and how the model is trained (e.g. tag)
    simple example: True and not False or True and not True and not False : True
    adding hierarchy: ((True and not False) or (True and True)) and False : False
    adding variables: a=10; b=5; a>5 and b>2 : True
    more than one output: a=10; b=5; a>5 and b>2: True, a<5 and b>2 : False, a>=10 and b>2 : True
    mathematical expressions: Regression task
    """

    # todo include min max to truncate operations in between lengths
    # todo: attach several of these and generate multiple output
    assert dtype in LOGIC_TYPES

    if dtype == 'bool_simple':
        grammar = CFG.fromstring(BOOL_CFG_SIMPLE)
        inps = random.sample(list(generate(grammar, depth=m)), n)  # depth=5, num=182712, depth=4, num=302
        pythonizer = lambda x: eval(' '.join(x))
        outs = [pythonizer(inp) for inp in inps]

    elif dtype == 'bool':
        grammar = CFG.fromstring(BOOL_CFG)
        inps = random.sample(list(generate(grammar, depth=m)), n)  # depth=5, num=357014, depth=4, num=422
        pythonizer = lambda x: eval(' '.join(x))
        outs = [pythonizer(inp) for inp in inps]

    # elif dtype == 'var_bool':
        # grammar = CFG.fromstring(BOOL_CFG)
        # inps = random.sample(list(generate(grammar, depth=m)), n)
        # inps = pd.DataFrame([' '.join(sent) for sent in inps])
        # inps = inps.astype(str)
        # pythonizer = lambda x: eval(f"{x}")  # todo: check if we need formatting
        # outs = inps.applymap(pythonizer)

    elif dtype in ['algebraic_int', 'algebraic_float']:
        cfg = ARITH_CFG_INT if dtype == 'algebraic_int' else ARITH_CFG  # if floats > wide range of outs!
        grammar = CFG.fromstring(cfg)
        all = list(generate(grammar, depth=m))
        # trunc = [a for a in all if len(a) == m]
        trunc = all
        inps = random.sample(trunc, n)  # depth=5, num=16836
        inps = [set_value(inp) for inp in inps]
        pythonizer = lambda x: eval(' '.join(x))
        outs = [pythonizer(inp) for inp in inps]

    elif dtype in ['var+algebraic_int', 'var+algebraic_float']:
        cfg = ARITH_VAR_CFG_INT if dtype == 'var+algebraic_int' else ARITH_VAR_CFG
        grammar = CFG.fromstring(cfg)
        all = list(generate(grammar, depth=m))
        # trunc = [a for a in all if len(a) == m]
        trunc = all
        inps = random.sample(trunc, n)
        inps = [set_value_var(inp) for inp in inps]
        eval_var_func = lambda x: eval_var(' '.join(x))
        outs = [eval_var_func(inp) for inp in inps]

    elif dtype in ['var+comparative_int', 'var+comparative_float']:
        # defaults to two variables # todo: add more variables?
        inps = [gen_comp(m) for _ in range(n)]
        eval_var_func = lambda x: eval_comp(' '.join(x))
        outs = [eval_var_func(inp) for inp in inps]

    return inps, outs


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
    parser.add_argument('--fname', type=str)
    args = parser.parse_args()

    inps, outs = run_task(args)
    with open(args.fname, 'a', encoding='utf-8') as jl:
        vars(args).pop('fname')
        jl.write("{ \"config\": " + json.dumps(vars(args)) + ", \"data\": [")
        for j, (i, o) in enumerate(zip(inps, outs)):
            if j == args.n-1:
                jl.write(json.dumps(dict(language='java', inps=i, outs=o)))
            else:
                jl.write(json.dumps(dict(language='java', inps=i, outs=o)) + ", ")
        jl.write("]}\n")
