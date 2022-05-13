from ml4code_interp.featurizers.parser_utils import parse
from ml4code_interp.featurizers.global_features import GlobalFeaturesExtractor
from ml4code_interp.featurizers.token_labeler import TokenTypeLabeler, TokenLabel
from ml4code_interp.featurizers.DFG import DFGEdge, getDFG

class TestGlobalFeaturizer:
    def helper(self, code, lang='java'):
        return GlobalFeaturesExtractor(
            lang,
            parse(lang, code)
        ).featurize()

    def test_if(self, ):
        result = self.helper(
            'class FooTest { public void foo() { if (true) { System.out.println("Hello"); } } }',
        )
        assert result['has_if']
        assert not result['has_while']
        assert not result['has_for']

    def test_while(self, ):
        result = self.helper(
            'class FooTest { public void foo() { while (true) { System.out.println("Hello"); } } }',
        )
        assert result['has_while']
        assert not result['has_for']
        assert not result['has_if']

    def test_for(self, ):
        assert self.helper(
            'class FooTest { public void foo() { for (int i = 0; i < 10; i++) { System.out.println("Hello"); } } }',
        )['has_for']

    def test_switch(self, ):
        assert self.helper(
            'class FooTest { public void foo() { switch (1) { case 1: System.out.println("Hello"); } } }',
        )['has_switch']

    def test_throw(self, ):
        assert self.helper(
            'class FooTest { public void foo() { throw new Exception(); } }',
        )['has_throw']

    def test_function_call(self, ):
        assert self.helper(
            'class FooTest { public void foo() { System.out.println("Hello"); } }',
        )['has_invoke']

    def test_no_if(self, ):
        assert not self.helper(
            'class FooTest { public void foo() { System.out.println("Hello"); } }',
        )['has_if']

class TestTokenLabelFeaturizer:
    def test_token_label_featurizer_1(self):
        lang = 'java'
        code = """
    {
        int a = 10;
        int c = a;
    }
    """

        # TokenLabel(token=Token(idx:0;loc:((1, 0), (1, 1))       type:{  text:{), label='{')
        # TokenLabel(token=Token(idx:1;loc:((2, 4), (2, 7))       type:int        text:int), label='int')
        # TokenLabel(token=Token(idx:2;loc:((2, 8), (2, 9))       type:identifier text:a), label='identifier')
        # TokenLabel(token=Token(idx:3;loc:((2, 10), (2, 11))     type:=  text:=), label='=')
        # TokenLabel(token=Token(idx:4;loc:((2, 12), (2, 14))     type:decimal_integer_literal    text:10), label='decimal_integer_literal')
        # TokenLabel(token=Token(idx:5;loc:((2, 14), (2, 15))     type:;  text:;), label=';')
        # TokenLabel(token=Token(idx:6;loc:((3, 4), (3, 7))       type:int        text:int), label='int')
        # TokenLabel(token=Token(idx:7;loc:((3, 8), (3, 9))       type:identifier text:c), label='identifier')
        # TokenLabel(token=Token(idx:8;loc:((3, 10), (3, 11))     type:=  text:=), label='=')
        # TokenLabel(token=Token(idx:9;loc:((3, 12), (3, 13))     type:identifier text:a), label='identifier')
        # TokenLabel(token=Token(idx:10;loc:((3, 13), (3, 14))    type:;  text:;), label=';')
        # TokenLabel(token=Token(idx:11;loc:((4, 0), (4, 1))      type:}  text:}), label='}')

        parse_result = parse(lang, code)
        token_label_featurizer = TokenTypeLabeler(lang, parse_result)
        result = token_label_featurizer.featurize()

        assert len(result) == 12
        assert result[0] == TokenLabel(token=parse_result.toks[0], label='{')
        assert result[1] == TokenLabel(token=parse_result.toks[1], label='int')
        assert result[2] == TokenLabel(token=parse_result.toks[2], label='identifier')
        assert result[3] == TokenLabel(token=parse_result.toks[3], label='=')
        assert result[4] == TokenLabel(token=parse_result.toks[4], label='decimal_integer_literal')
        assert result[5] == TokenLabel(token=parse_result.toks[5], label=';')
        assert result[6] == TokenLabel(token=parse_result.toks[6], label='int')
        assert result[7] == TokenLabel(token=parse_result.toks[7], label='identifier')
        assert result[8] == TokenLabel(token=parse_result.toks[8], label='=')
        assert result[9] == TokenLabel(token=parse_result.toks[9], label='identifier')
        assert result[10] == TokenLabel(token=parse_result.toks[10], label=';')
        assert result[11] == TokenLabel(token=parse_result.toks[11], label='}')

class TestDFGFeaturizer:
    def test_dfg_1(self):
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
        DFG_edges = getDFG(lang, parse_result)

        assert len(DFG_edges) == 7
        # DFGEdge(src=Token(idx:31;loc:((3, 12), (3, 14)) type:decimal_integer_literal    text:10), dst=Token(idx:29;loc:((3, 8), (3, 9)) type:identifier text:a))
        # DFGEdge(src=Token(idx:29;loc:((3, 8), (3, 9))   type:identifier text:a), dst=Token(idx:35;loc:((4, 11), (4, 12))        type:identifier text:a))
        # DFGEdge(src=Token(idx:40;loc:((5, 8), (5, 9))   type:identifier text:a), dst=Token(idx:35;loc:((4, 11), (4, 12))        type:identifier text:a))
        # DFGEdge(src=Token(idx:42;loc:((5, 12), (5, 13)) type:identifier text:a), dst=Token(idx:40;loc:((5, 8), (5, 9))  type:identifier text:a))
        # DFGEdge(src=Token(idx:44;loc:((5, 16), (5, 17)) type:decimal_integer_literal    text:1), dst=Token(idx:40;loc:((5, 8), (5, 9))  type:identifier text:a))
        # DFGEdge(src=Token(idx:29;loc:((3, 8), (3, 9))   type:identifier text:a), dst=Token(idx:42;loc:((5, 12), (5, 13))        type:identifier text:a))
        # DFGEdge(src=Token(idx:40;loc:((5, 8), (5, 9))   type:identifier text:a), dst=Token(idx:42;loc:((5, 12), (5, 13))        type:identifier text:a))
        assert DFG_edges[0] == DFGEdge(parse_result.toks[31], parse_result.toks[29])
        assert DFG_edges[1] == DFGEdge(parse_result.toks[29], parse_result.toks[35])
        assert DFG_edges[2] == DFGEdge(parse_result.toks[40], parse_result.toks[35])
        assert DFG_edges[3] == DFGEdge(parse_result.toks[42], parse_result.toks[40])
        assert DFG_edges[4] == DFGEdge(parse_result.toks[44], parse_result.toks[40])
        assert DFG_edges[5] == DFGEdge(parse_result.toks[29], parse_result.toks[42])
        assert DFG_edges[6] == DFGEdge(parse_result.toks[40], parse_result.toks[42])
