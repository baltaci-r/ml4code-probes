from pytest import fixture
from ml4code_interp.featurizers.global_features import GlobalFeaturesExtractor
from ml4code_interp.featurizers.parser_utils import parse
from ml4code_interp.models.base import InterpretableModel
from ml4code_interp.tasks.global_feature_pred import GlobalFeaturePredTask
from ml4code_interp.tasks.token_label_pred import TokenLabelPredTask

@fixture(scope='session')
def model() -> InterpretableModel:
    return InterpretableModel("microsoft/codebert-base")

def test_global_feature_pred_task(model):
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
    num_layers = 1 + len(model.model.encoder.layer)
    num_emb_features = model.model.encoder.layer[0].output.dense.out_features

    prepared_data = GlobalFeaturePredTask(model, [(lang, code)]).prepare_data()

    assert list(prepared_data.keys()) == ['embeds', 'features']
    assert set(prepared_data['features'].keys()) == set(GlobalFeaturesExtractor.FEATURE_NAMES)

    assert len(prepared_data['embeds']) == 1
    assert len(prepared_data['embeds'][0]) == num_layers

    parse_result = parse(lang, code)
    tokenized = model._prepare_input(parse_result)
    num_model_toks = len(tokenized['input_ids'][0])
    assert all(emb.shape[0] == num_model_toks for emb in prepared_data['embeds'][0])
    assert all(emb.shape[1] == num_emb_features for emb in prepared_data['embeds'][0])

def test_token_label_task_pred(model):
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
    num_layers = 1 + len(model.model.encoder.layer)
    num_emb_features = model.model.encoder.layer[0].output.dense.out_features

    parse_result = parse(lang, code)
    tokenized = model._prepare_input(parse_result)
    num_model_toks = len(tokenized['input_ids'][0])

    prepared_data = TokenLabelPredTask(model, [(lang, code)]).prepare_data()

    assert list(prepared_data.keys()) == ['embeds', 'labels']
    
    assert len(prepared_data['embeds']) == 1
    assert len(prepared_data['embeds'][0]) == num_layers
    assert all(len(L) == len(parse_result.toks) for L in prepared_data['embeds'][0])
    
    # assert all(emb.shape[0] == num_model_toks for emb in prepared_data['embeds'][0][0])
    assert all(emb.shape[1] == num_emb_features for emb in prepared_data['embeds'][0][0])
    