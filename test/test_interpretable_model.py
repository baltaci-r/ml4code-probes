from pytest import fixture

from ml4code_interp.featurizers.parser_utils import parse
from ml4code_interp.models.base import InterpretableModel

@fixture(scope='session')
def model() -> InterpretableModel:
    return InterpretableModel("microsoft/codebert-base")

def test_alignment_1(model: InterpretableModel):
    code = """public void reduce(UTF8 key, Iterator<UTF8> values,
                        OutputCollector<UTF8, UTF8> output, Reporter reporter) throws IOException 
    {
        int a = 10;
        while (a > 0) {
            a = a - 1;
        }
    }
    """
    parse_result = parse('java', code)
    tokenized = model._prepare_input(parse_result)

    alignment = model._align_tokens(parse_result, tokenized)
    assert len(alignment) == len(parse_result.toks)
    model_tok_idxs = [idx for model_tok_idxs_local in alignment for idx in model_tok_idxs_local]
    # if we look at model_tok_idxs from alignment, we'll ignore all whitespace-ish tokens.
    # so we've to be careful while doing string comparison. We decode the model_tok_idxs tokens, and remove spaces
    # and compare with original code without any whitespaces. But original code after joining may still have space if there are comments.
    assert model.tokenizer.decode(tokenized['input_ids'][0][model_tok_idxs]).replace(' ', '') == ''.join(tok.str for tok in parse_result.toks).replace(' ', '')

    for tok, model_tok_idxs in zip(parse_result.toks, alignment):
        if len(tok.str) == 0:
            assert tok.char_start == tok.char_end
            assert model_tok_idxs == []
        else:
            assert tokenized.token_to_chars(model_tok_idxs[0]).start == tok.char_start
            assert tokenized.token_to_chars(model_tok_idxs[-1]).end == tok.char_end

def test_get_embeddings_1(model: InterpretableModel):
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
    embedding_dim = model.model.encoder.layer[0].output.dense.out_features

    parse_result = parse('java', code)
    embeddings = model.get_embeddings(parse_result)

    assert len(embeddings) == num_layers

    tokenized = model._prepare_input(parse_result)
    alignment = model._align_tokens(parse_result, tokenized)
    
    assert len(alignment) == len(parse_result.toks)
    for l in embeddings:
        assert l.shape[0] == tokenized['input_ids'][0].shape[0]
        assert l.shape[1] == embedding_dim
        
def test_get_embeddings_per_token_1(model: InterpretableModel):
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
    embedding_dim = model.model.encoder.layer[0].output.dense.out_features

    parse_result = parse('java', code)
    embeddings = model.get_embeddings_per_token(parse_result)

    assert len(embeddings) == num_layers

    tokenized = model._prepare_input(parse_result)
    alignment = model._align_tokens(parse_result, tokenized)
    
    assert len(alignment) == len(parse_result.toks)
    for l in embeddings:
        assert len(l) == len(parse_result.toks)
        for tok_idx, tok_emb in enumerate(l):
            assert tok_emb.shape[0] == len(alignment[tok_idx])
            assert tok_emb.shape[1] == embedding_dim
