"""
This module provides a featurizer that checks if a method has certain statements/structures

e.g., whether the code has an if statement, a while statement, etc.

The input is a tree-sitter AST's root node.
"""


FEATURE_NAMES = dict(
    HAS_IF = 0,
    HAS_WHILE = 1,
    HAS_FOR = 2,
    HAS_SWITCH = 3,
    HAS_TRY_CATCH = 4,
    HAS_THROW = 5,
    HAS_INVOKE = 6,
)


class GlobalFeaturesExtractor(object):
    def __init__(self, lang, parse_result):
        self.root_node = parse_result.tree.root_node
        self.features = [0 for _ in FEATURE_NAMES]

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
            self.features[FEATURE_NAMES['HAS_IF']]        |= node.type in self.lang_2_elem_map['if']
            self.features[FEATURE_NAMES['HAS_WHILE']]     |= node.type in self.lang_2_elem_map['while']
            self.features[FEATURE_NAMES['HAS_FOR']]       |= node.type in self.lang_2_elem_map['for']
            self.features[FEATURE_NAMES['HAS_SWITCH']]    |= node.type in self.lang_2_elem_map['switch']
            self.features[FEATURE_NAMES['HAS_TRY_CATCH']] |= node.type in self.lang_2_elem_map['try_catch']
            self.features[FEATURE_NAMES['HAS_THROW']]     |= node.type in self.lang_2_elem_map['throw']
            self.features[FEATURE_NAMES['HAS_INVOKE']]    |= node.type in self.lang_2_elem_map['invoke']
            for child in node.children:
                q.append(child)

    def featurize(self):
        return self.features