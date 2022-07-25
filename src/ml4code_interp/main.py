import json

import pandas as pd
import torch
import os
from pathlib import Path
import argparse
import sys
sys.path.append('./')
from ml4code_interp.featurizers.parser_utils import parse
from ml4code_interp.featurizers.DFG import getDFG
# from ml4code_interp.featurizers.global_features import GlobalFeaturesExtractor
from ml4code_interp.featurizers.token_labeler import TokenTypeLabeler

from ml4code_interp.models.base import InterpretableModel
from ml4code_interp.tasks.global_feature_pred import GlobalFeaturePredTask
from ml4code_interp.tasks.token_label_pred import TokenLabelPredTask
from ml4code_interp.tasks.numeracy import NumeracyTask

# data = [
#     ('java', 'public class Test { public static void main(String[] args) { System.out.println("Hello World!"); } }'),
#     ('java', """public void reduce(UTF8 key, Iterator<UTF8> values,
#                     OutputCollector<UTF8, UTF8> output, Reporter reporter) throws IOException 
# {
#     int a = 10;
#     while (a > 0) {
#         a = a - 1;
#     }
# }
# """)
# ]

def get_task(model, task):
    return {
        'global_feature': GlobalFeaturePredTask,
        'token_label': TokenLabelPredTask,
        'numeracy': NumeracyTask,
    }[task](model)


def main(args):
    model = InterpretableModel(args.model_name)
    task = get_task(model, args.task)
    data = task.load_data(args.path, args.data_path)
    results = task.run(data, args.num)
    # fixme: decoding should only be one token, not sure why we are getting several tokens
    #  the addition we need to pad, words not working getting split as tokens, all other
    #  are supposed to be in lstm and then we pad!
    with open(args.results_path, "w") as f:
        json.dump(results, f)


# embeddings_per_token = model.get_embeddings_per_token(parse_result)
# for tok, x in zip(parse_result.toks, embeddings_per_token):
#     print(tok.str, len(x), x[0].shape if len(x) > 0 else None)

# print(prepared_data)

# print(len(prepared_data['embeds'][0]), prepared_data['embeds'][0][0].shape)

# data_path = "/home/parthdt2/project/interpretability/data/CodeSearchNet/resources/data/java/final/jsonl/test/java_test_0.jsonl"
# model_name = "microsoft/codebert-base"

# # tokenized = model._prepare_input(parse_result)
# # alignment = model._align_tokens(parse_result, tokenized)
# # for tok, x in zip(parse_result.toks, alignment):
# #     print(tok.str, x)

# # TODO: generate datasets
# # Two types of datasets: GlobalFeatureDataset and TokenFeatureDataset
# # GlobalFeatureDataset:
# #   one column per feature. column name is feature name, cell value is feature value
# #   one row per code input.
# # TokenFeatureDataset:
# #   columns: doc_id, token_text, char_start, char_end, token_label -- note that token_text is optional, only for debugging
# #   one row per token per code input.
# # This should be merged with the model's tokenizer's [token_idx] -- mapping OUR token indices to the tokenizer's token indices

# # Once the data is ready, we can obtain the model's embeddings of the tokens, and use that to train the classifier.

def update_args(args):
    cur_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    # args.path = cur_dir / f'../../probing_data/{args.task}/{args.subtask}/prepared_data_test.pt' if args.subtask else \
    #             cur_dir / f'../../probing_data/{args.task}/prepared_data_test.pt'
    data_dir = cur_dir / f'../../probing_data/{args.task}/{args.model_name}'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    args.path = cur_dir / f'../../probing_data/{args.task}/{args.model_name}/prepared_data_test.pt'
    args.results_path = cur_dir / f"../../results/{args.task}_{args.model_name.replace('/', '_')}_{args.num}.json"
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="microsoft/codebert-base")
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--task', type=str, choices=['global_feature', 'token_label', 'numeracy'])
    parser.add_argument('--num', type=int, help='Number of probing samples')
    args = parser.parse_args()
    args = update_args(args)

    main(args)