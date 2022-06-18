import json
import torch
import os
from pathlib import Path

from ml4code_interp.featurizers.parser_utils import parse
from ml4code_interp.featurizers.DFG import getDFG
# from ml4code_interp.featurizers.global_features import GlobalFeaturesExtractor
from ml4code_interp.featurizers.token_labeler import TokenTypeLabeler

from ml4code_interp.models.base import InterpretableModel
from ml4code_interp.tasks.global_feature_pred import GlobalFeaturePredTask

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

def read_raw_data(jsonl_path):
    ctr = 0
    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            lang = data['language']
            code = data['code']
            if lang == "java":
                code = "public class Test {\n" + code + "\n}" 
            else:
                raise "Unsupported language"
            
            yield lang, code
            ctr += 1
            if ctr > 1000:
                return

# parse_result = parse(lang, code)

data_path = "/home/parthdt2/project/interpretability/data/CodeSearchNet/resources/data/java/final/jsonl/test/java_test_0.jsonl"
model_name = "microsoft/codebert-base"
model = InterpretableModel(model_name)
# embeddings_per_token = model.get_embeddings_per_token(parse_result)
# for tok, x in zip(parse_result.toks, embeddings_per_token):
#     print(tok.str, len(x), x[0].shape if len(x) > 0 else None)

cur_dir = Path(os.path.dirname(os.path.abspath(__file__)))
path = cur_dir/'../../probing_data/global_feature_pred_task/prepared_data_test.pt'

task = GlobalFeaturePredTask(model)
# prepared_data = task.prepare_data(read_raw_data(data_path))
# torch.save(prepared_data, path)
# os.exit()

data = torch.load(path)
results = task.run(data)

with open(cur_dir/f"../../results/global_pred_{model_name.replace('/', '_')}.json", "w") as f:
    json.dump(results, f)

# print(prepared_data)

# print(len(prepared_data['embeds'][0]), prepared_data['embeds'][0][0].shape)

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