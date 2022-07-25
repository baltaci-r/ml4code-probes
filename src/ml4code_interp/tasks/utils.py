import json


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