"""
This module provides a featurizer that checks if a method has certain statements/structures

e.g., whether the code has an if statement, a while statement, etc.

The input is a tree-sitter AST's root node.
"""

class GlobalFeaturesExtractor(object):
    FEATURE_NAMES = [
        'has_if',
        'has_while',
        'has_for',
        'has_switch',
        'has_try_catch',
        'has_throw',
        'has_invoke',
    ]
    def __init__(self, lang, parse_result):
        self.root_node = parse_result.tree.root_node
        self.features = {f: False for f in self.FEATURE_NAMES}

        self.lang_2_elem_map = self._load_lang_2_elem_map(lang)

        self._bst()
    
    def _load_lang_2_elem_map(self, lang):
        return {
            'java': {
                'if': ['if_statement'],
                'while': ['while_statement'],
                'for': ['for_statement', 'enhanced_for_statement'],
                'switch': ['switch_expression'],
                'try_catch': ['try_statement'],
                'throw': ['throw_statement'],
                'invoke': ['method_invocation']
            }
        }[lang]

    def _bst(self):
        q = [self.root_node]
        while len(q) > 0:
            node = q.pop()
            self.features['has_if']        |= node.type in self.lang_2_elem_map['if']
            self.features['has_while']     |= node.type in self.lang_2_elem_map['while']
            self.features['has_for']       |= node.type in self.lang_2_elem_map['for']
            self.features['has_switch']    |= node.type in self.lang_2_elem_map['switch']
            self.features['has_try_catch'] |= node.type in self.lang_2_elem_map['try_catch']
            self.features['has_throw']     |= node.type in self.lang_2_elem_map['throw']
            self.features['has_invoke']    |= node.type in self.lang_2_elem_map['invoke']
            for child in node.children:
                q.append(child)

    def featurize(self):
        return self.features