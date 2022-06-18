"""
This module contains the code for the global feature prediction task.
"""

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from typing import List, Tuple
from tqdm import tqdm

import numpy as np
import torch
from ml4code_interp.featurizers.global_features import FEATURE_NAMES, GlobalFeaturesExtractor

from ml4code_interp.featurizers.parser_utils import parse
from ml4code_interp.models.base import InterpretableModel

class GlobalFeaturePredTask:
    """
    This class implements the global feature prediction task.
    """

    def __init__(self, model: InterpretableModel):
        """
        Initialize the task.

        :param model: The model to use.
        :param data: A list of tuples of (lang, code).
        """
        self.model = model

    def prepare_data(self, raw_data: List[Tuple[str, str]]) -> dict:
        output = {
            "feature_names": FEATURE_NAMES,
            "lang":         [],  # type: List[str]
            "code":         [],  # type: str[NumDocs]
            "tokens":       [],  # type: List[List[str]] -- tokenized code
            "embeds":       [],  # type: Tensor[NumLayers][NumDocs] where Tensor is of shape (NumModelTokens, EmbeddingDim).
                                 #       Note that NumModelTokens may be different for every tensor.
            "features":     [[] for _ in FEATURE_NAMES] # type: List[List[int]]
            # ~~~~Examples~~~~:
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

        num_skipped = 0
        num_layers = None
        for i, (lang, code) in enumerate(raw_data):
            # TODO: batching
            parse_result = parse(lang, code)
            try:
                embeddings_by_layer = self.model.get_embeddings(parse_result)
            except:
                print(f"Skipping entry {i}") # likely because the code is too long.
                num_skipped += 1
            
            features = GlobalFeaturesExtractor(lang, parse_result).featurize()

            if num_layers is None:
                num_layers = len(embeddings_by_layer)
                output['embeds'] = [[] for _ in range(num_layers)]
            
            for l in range(num_layers):
                output['embeds'][l].append(embeddings_by_layer[l])

            output['lang'].append(lang)
            output['code'].append(parse_result.code)
            output['tokens'].append([t.str for t in parse_result.toks])
            for i, f in enumerate(features):
                output['features'][i].append(f)

        print(f"Skipped {num_skipped} entries.")
        return output

    def run(self, data):
        """
        Run the task.
        """
        # type - Tensor[NumLayers][NumDocs] where Tensor is of shape (NumModelTokens, EmbeddingDim).
        inputs = data['embeds']
        num_layers = len(inputs)

        for f_idx, f_name in tqdm(enumerate(data['feature_names'])):
            y = data['features'][f_idx]
            
            for l in tqdm(range(num_layers)):
                # type - Tensor[NumDocs], where Tensor is of shape (EmbeddingDim)
                X = []
                for row in inputs[l]:
                    X.append(row.sum(dim=0).numpy()) # add up embeddings of every token in the doc.

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.5, random_state=42
                )
                if len(set(y_train)) == 1:
                    print(f"Skipping {f_name} because all values are the same.")
                    continue
                
                pipe = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(class_weight='balanced', random_state=0),
                )
                pipe.fit(X_train, y_train)
                # y_test = [1 for _ in y_test]
                acc = pipe.score(X_test, y_test)
                print(f"{f_name} layer {l}: {acc}")

                y_pred = pipe.predict(X_test)
                print(confusion_matrix(y_test, y_pred))
                print()
            
            print("=" * 80)