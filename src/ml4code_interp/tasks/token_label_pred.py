"""
This module contains the code for the token label prediction task.
"""

from typing import List, Tuple

from featurizers.token_labeler import TokenTypeLabeler
from featurizers.parser_utils import parse
from models.base import InterpretableModel

class TokenLabelPredTask:
    """
    This class implements the token label prediction task.
    """

    def __init__(self, model: InterpretableModel, data: List[Tuple[str, str]]):
        """
        Initialize the task.

        :param model: The model to use.
        :param data: A list of tuples of (lang, code).
        """
        self.model = model
        self.data = data

    def prepare_data(self):
        output = {
            # "doc_id": [],      # type: int[NumDocs]
            "code":         [],  # type: str[NumDocs]
            "embeds":       [],  # type: Tensor[NumDocs][NumLayers][NumTokens] where Tensor is of shape (NumModelTokens, EmbeddingDim).
                                 #       Note that NumModelTokens may be different for every tensor.
            "labels":       []   # type: str[NumDocs][NumTokens]
        }

        for lang, code in self.data:
            # TODO: batching
            parse_result = parse(lang, code)
            embeddings = self.model.get_embeddings_per_token(parse_result)
            labels = TokenTypeLabeler(lang, parse_result).featurize()
            
            output['code'].append(code)
            output['embeds'].append(embeddings)
            output['labels'].append(labels)
        
        return output

    def run(self):
        """
        Run the task.
        """
        pass