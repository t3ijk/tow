
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
    with open('tokenizer_config.json', 'w') as f:
        f.write(json.dumps(tk.get_config(), indent=4))

test_tokenizer('qwew<pad>qeqwewqe</s>qwewqeqw<unk>ewqe')
test_tokenizer('hello world!')
test_tokenizer('你好世界！')
test_tokenizer('你好世界！hello world! \n1 \t !@#$%^&* <<<<extra_id_119> 123456')