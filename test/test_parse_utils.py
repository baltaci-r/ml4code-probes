from ml4code_interp.featurizers.parser_utils import parse

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

def test_char_idx_mapping():
    parse_result = parse(lang, code)
    for tok in parse_result.toks:
        # print(tok, tok.char_start, tok.char_end)
        assert tok.str == code[tok.char_start:tok.char_end], (tok.str, code[tok.char_start:tok.char_end])

    code2 = "\n" + code
    parse_result = parse(lang, code2)
    for tok in parse_result.toks:
        # print(tok, tok.char_start, tok.char_end)
        assert tok.str == code2[tok.char_start:tok.char_end], (tok.str, code2[tok.char_start:tok.char_end])
