from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModel, BatchEncoding
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from ml4code_interp.featurizers.parser_utils import ParseResult

class InterpretableModel(object):
    """
    Base class for all interpretable models.
    """

    def __init__(self, model_name_or_path: str):
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        assert self.tokenizer.is_fast, "We only support fast tokenizers for now."
    
    def get_embeddings_per_token(self, parse_result: ParseResult) -> List[List[torch.Tensor]]:
        """
        Returns layer-wise embeddings for each ParsedToken. Since a ParsedToken may have multiple embeddings,
        because a ParsedToken may be split into multiple ModelTokens, the final shape of the output is

        [num_layers][num_parsed_toks] -> Tensor(num_model_toks, embedding_size)
        Note that num_model_toks may be different for each ParsedToken. That is, each tensor may be of different dim0, but same dim1.
        
        It is up to the caller to aggregate the ModelToken embeddings into a single embedding.

        Note also that this doesn't include embeddings for whitespaces.
        """
        inputs, hidden_outputs = self._infer(parse_result)
        alignment_map = self._align_tokens(parse_result, inputs)

        result = []
        for layer_out in hidden_outputs:
            result.append([])
            for model_tok_idxs in alignment_map:
                result[-1].append(layer_out[model_tok_idxs, :])
        return result
    
    def get_embeddings(self, parse_result: ParseResult) -> List[torch.Tensor]:
        """
        Same as get_embeddings_per_token, except this doesn't group embeddings of a ParsedToken
        Also, this includes embeddings for whitespaces, whereas get_embeddings_per_token doesn't.

        :return: List[Tensor] where tensor is of shape (num_model_toks, embedding_size) and len(result) == num_layers
        """
        _, hidden_outputs = self._infer(parse_result)
        return hidden_outputs
    
    def _infer(self, parse_result) -> Tuple[BatchEncoding, BaseModelOutputWithPoolingAndCrossAttentions]:
        inputs = self._prepare_input(parse_result)
        
        with torch.no_grad():
            model_output = self.model(**inputs, output_hidden_states=True)
        
        hidden_outputs = [out.squeeze(0) for out in model_output.hidden_states] # list of (num_toks, hidden_size), len(result) == num_layers
        return inputs, hidden_outputs

    def _prepare_input(self, parse_result: ParseResult) -> BatchEncoding:
        """
        Prepares the input for the model.
        Returns a list of tensors.

        Override if the model needs special input.
        """
        return self.tokenizer(parse_result.code, return_tensors='pt')
    
    def _align_tokens(
        self, parse_result: ParseResult, tokenized_output: BatchEncoding
    ) -> List[List[int]]:
        """
        Returns a list of lists of token indices.
        Each entry in the list corresponds to a ParsedToken.
        Each entry in the inner list corresponds to a ModelToken.
        """
        result = []
        for tok in parse_result.toks:
            char_start, char_end = tok.char_start, tok.char_end
            model_toks = sorted(set([
                tokenized_output.char_to_token(i) for i in range(char_start, char_end)
            ]))
            result.append(model_toks)
        return result
