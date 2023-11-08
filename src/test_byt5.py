
from model_byt5.tokenizer import text2tokens, tokens2ids, ids2tokens, tokens2text, text_remove_split_tokens

def test_tokenizer(input):
    print('--------------------------------')
    print(input)
    tokens = text2tokens(input)
    print(tokens)
    ids = tokens2ids(tokens)
    print(ids)
    tokens2 = ids2tokens(ids)
    print(tokens2)

    if tokens != tokens2:
        raise Exception(tokens, tokens2)    

    output = tokens2text(tokens2)
    # print(output)    

    input_cleaned = ''.join(text_remove_split_tokens(input))
    print(input_cleaned)
    if input_cleaned != output:
        raise Exception(input_cleaned, output)


test_tokenizer('qwew<pad>qeqwewqe</s>qwewqeqw<unk>ewqe')
test_tokenizer('hello world!')
test_tokenizer('你好世界！')
test_tokenizer('wqwq的武器武器大全气得我的武器33 我去打网球的武器无穷大无穷大-3243253-=121325434365366')


