"""
This module contains the code for the global feature prediction task.
"""

from typing import List, Tuple
from ml4code_interp.featurizers.global_features import GlobalFeaturesExtractor

from ml4code_interp.featurizers.parser_utils import parse
from ml4code_interp.models.base import InterpretableModel

class GlobalFeaturePredTask:
    """
    This class implements the global feature prediction task.
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
            # "token_texts":  [],# type: str[NumDocs][NumTokens]
            "embeds":       [],  # type: Tensor[NumDocs][NumLayers] where Tensor is of shape (NumModelTokens, EmbeddingDim).
                                 #       Note that NumModelTokens may be different for every tensor.
            "features":     {f: [] for f in GlobalFeaturesExtractor.FEATURE_NAMES}
            # Examples:
            # {
            #    "has_if":           [], # type: bool[NumDocs]
            #    "has_while":        [], # type: bool[NumDocs]
            #    "has_for":          [], # type: bool[NumDocs]
            #    "has_switch":       [], # type: bool[NumDocs]
            #    "has_try_catch":    [], # type: bool[NumDocs]
            #    "has_throw":        [], # type: bool[NumDocs]
            #    "has_invoke":       [], # type: bool[NumDocs]
            #    "cyclomatic_complexity": [], # type: int[NumDocs]
            # }
        }

        for lang, code in self.data:
            # TODO: batching
            parse_result = parse(lang, code)
            embeddings = self.model.get_embeddings(parse_result)
            features = GlobalFeaturesExtractor(lang, parse_result).featurize()
            
            # output['token_texts'].append([tok.str for tok in parse_result.toks])
            output['embeds'].append(embeddings)
            for k, v in features.items():
                output['features'][k].append(v)
        
        return output

    def run(self):
        """
        Run the task.
        """
        pass