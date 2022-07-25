"""
Collection of featurizers that labels each token
"""

from dataclasses import dataclass

from featurizers.parser_utils import ParseResult, ParsedToken

@dataclass
class TokenLabel:
    token: ParsedToken # use this to get the token position (char_start and char_end)
    label: str

class TokenTypeLabeler:
    """
    Labels each token with its type
    """
    def __init__(self, lang, parse_result: ParseResult):
        self.parse_result = parse_result
    
    def featurize(self):
        # TODO: convert token type to generic labels like 'keyword', 'identifier' etc.
        return [TokenLabel(token, token.node.type) for token in self.parse_result.toks]
