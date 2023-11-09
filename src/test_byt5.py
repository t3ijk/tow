
from model_byt5.tokenizer import Byt5Tokenizer
import json
def test_tokenizer(input):
    tk = Byt5Tokenizer()
    print('--------------------------------')
    print(input)
    tokens = tk.text2tokens(input)
    print(tokens)
    ids = tk.tokens2ids(tokens)
    print(ids)
    tokens2 = tk.ids2tokens(ids)
    print(tokens2)

    if tokens != tokens2:
        raise Exception(tokens, tokens2)    

    output = tk.tokens2text(tokens2)
    # print(output)    

    input_cleaned = ''.join(tk.text_clean_special_tokens(input))
    print(input_cleaned)
    if input_cleaned != output:
        raise Exception(input_cleaned, output)
    with open('tokenizer_tokens.json', 'w') as f:
        f.write(json.dumps(tk.all_tokens, indent=4))
    json.dumps(tk.all_tokens)    

test_tokenizer('qwew<pad>qeqwewqe</s>qwewqeqw<unk>ewqe')
test_tokenizer('hello world!')
test_tokenizer('你好世界！')
test_tokenizer('wqwq的武器武器大全气得我的武器33 我去打网球的武器无穷大无穷大-3243253-=121325434365366')


