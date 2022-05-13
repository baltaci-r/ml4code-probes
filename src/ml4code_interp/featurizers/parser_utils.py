from dataclasses import dataclass
import re
from io import StringIO
import  tokenize
from typing import Dict, List
import numpy as np
import tree_sitter
import os.path
def remove_comments_and_docstrings(source,lang):
    return source # don't use this
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
            # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp=[]
        for x in out.split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " " # note: a space and not an empty string
            else:
                return s
        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp=[]
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)

def get_loc_for_toks(root_node):
    if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='comment':
        # if it's a leaf node, I think.
        return [((root_node.start_point, root_node.end_point), root_node)]
    else:
        code_tokens=[]
        for child in root_node.children:
            code_tokens+=get_loc_for_toks(child)
        return code_tokens

def tree_to_variable_index(root_node,index_to_code):
    if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='comment':
        index=(root_node.start_point,root_node.end_point)
        _,code=index_to_code[index]
        if root_node.type!=code:
            # that's a way of checking if the node is a named node (I think)
            # https://tree-sitter.github.io/tree-sitter/using-parsers#named-vs-anonymous-nodes
            return [(root_node.start_point,root_node.end_point)]
        else:
            return []
    else:
        code_tokens=[]
        for child in root_node.children:
            code_tokens+=tree_to_variable_index(child,index_to_code)
        return code_tokens    

def loc_to_code_token(loc,code):
    """
    `code` is an array of lines of code
    `loc` is a tuple of (start_line,start_col) and (end_line,end_col)
        
    returns the token(s) corresponding to the loc.
    """
    start_point=loc[0]
    end_point=loc[1]
    if start_point[0]==end_point[0]:
        s=code[start_point[0]][start_point[1]:end_point[1]]
    else:
        s=""
        s+=code[start_point[0]][start_point[1]:]
        for i in range(start_point[0]+1,end_point[0]):
            s+=code[i]
        s+=code[end_point[0]][:end_point[1]]   
    return s

from tree_sitter import Language, Parser
supported_langs = [
    'python',
    'java',
]

parsers={}
for lang in supported_langs:
    try:
        LANGUAGE = Language(os.path.join(os.path.abspath(''), 'src/ml4code_interp/parser/my-languages.so'), lang)
    except:
        try:
            LANGUAGE = Language(os.path.join(os.path.abspath(''), 'ml4code_interp/parser/my-languages.so'), lang)
        except:
            raise

    parser = Parser()
    parser.set_language(LANGUAGE)
    parsers[lang]= parser

@dataclass
class ParsedToken:
    idx: int
    str: str
    node: tree_sitter.Node
    char_start: int
    char_end: int

    def __init__(self, idx, node, char_start, char_end):
        self.idx = idx
        self.str = node.text.decode("utf-8")
        self.node = node
        self.char_start = char_start
        self.char_end = char_end

    @property
    def loc(self):
        return (self.node.start_point, self.node.end_point)

    def __str__(self) : return f"Token(idx:{self.idx};loc:{self.loc}\ttype:{self.node.type}\ttext:{self.str})" #return self.str
    def __repr__(self): return f"Token(idx:{self.idx};loc:{self.loc}\ttype:{self.node.type}\ttext:{self.str})"

@dataclass
class ParseResult:
    code: str
    tree: object
    loc2tok: Dict[tuple, ParsedToken]  # map ( (start_line,start_col), (end_line,end_col) ) => Token
    loc2tok_legacy: Dict[tuple, tuple] # map ( (start_line,start_col), (end_line,end_col) ) => (idx,str)
    toks: List[ParsedToken]
    line_lengths_cumsum: List[int]

    #var2loc: Dict[str, tuple] # map variable name => ( (start_line,start_col), (end_line,end_col) )

def parse(lang, code):
    parser = parsers[lang]
    tree = parser.parse(bytes(code, "utf-8"))
    loc_tok_entries = sorted(get_loc_for_toks(tree.root_node)) # sort by location, so we can number the tokens
    line_lengths_cumsum = np.cumsum([len(line) for line in [''] + code.split('\n')])
    
    loc2tok = {}
    for idx, (loc, node) in enumerate(loc_tok_entries):
        start_char = int(loc[0][0]+line_lengths_cumsum[loc[0][0]] + loc[0][1])
        end_char = int(loc[1][0]+line_lengths_cumsum[loc[1][0]] + loc[1][1])
        loc2tok[loc] = ParsedToken(idx, node, start_char, end_char)
    
    loc2tok_legacy = {loc: (idx, node.text.decode('utf-8')) for idx, (loc, node) in enumerate(loc_tok_entries)}

    return ParseResult(
        code, tree, loc2tok, loc2tok_legacy, list(loc2tok.values()), line_lengths_cumsum
    )