import re

# 0-2 tokens
pad_token = "<pad>" # 0
eos_token = "</s>" # 1
unk_token = "<unk>" # 3

# vocabulary tokens, utf8 bytes, next 256
vocabulary_tokens = [chr(x) for x in range(256)]

# sentinel tokens, extra_ids, next 125
sentinel_tokens = [f"<extra_id_{x}>" for x in range(125)]

all_tokens = [unk_token, eos_token, pad_token, *vocabulary_tokens, *sentinel_tokens]
special_tokens = [unk_token, eos_token, pad_token, *sentinel_tokens]

def text2tokens(text):
    delimiters  = f"({'|'.join(special_tokens)})"
    
    results = []
    for str in re.split(delimiters , text):
        if str in special_tokens:
            results.append(str)
        else:
            encoded = str.encode('utf-8')
            results.extend([chr(code) for code in bytearray(encoded)])

    if results[-1] in [eos_token, unk_token, pad_token]:
        pass
    else:
        results.append(eos_token)

    return results

def tokens2ids(tokens):
    return [all_tokens.index(token) for token in tokens]

def ids2tokens(ids):
    return [all_tokens[ind] for ind in ids]

def tokens2text(tokens):
    filtered = filter(lambda token: token not in special_tokens, tokens)
    return bytearray([ord(c)  for c in list(filtered)]).decode("utf-8")

def text_clean_special_tokens(text):
    delimiters  = f"{'|'.join(special_tokens)}"
    return re.split(delimiters , text)
