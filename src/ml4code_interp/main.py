from ml4code_interp.featurizers.parser_utils import parse
from ml4code_interp.featurizers.DFG import getDFG
# from ml4code_interp.featurizers.global_features import GlobalFeaturesExtractor
from ml4code_interp.featurizers.token_labeler import TokenTypeLabeler

from ml4code_interp.models.base import InterpretableModel
from ml4code_interp.tasks.global_feature_pred import GlobalFeaturePredTask

lang = 'java'
code = """public void reduce(UTF8 key, Iterator<UTF8> values,
                    OutputCollector<UTF8, UTF8> output, Reporter reporter) throws IOException 
{
    int a = 10;
    while (a > 0) {
        a = a - 1;
    }
}
"""

parse_result = parse(lang, code)


# model = InterpretableModel("microsoft/codebert-base")
# # embeddings_per_token = model.get_embeddings_per_token(parse_result)
# # for tok, x in zip(parse_result.toks, embeddings_per_token):
# #     print(tok.str, len(x), x[0].shape if len(x) > 0 else None)

# prepared_data = GlobalFeaturePredTask(model, [('java', code)]).prepare_data()
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