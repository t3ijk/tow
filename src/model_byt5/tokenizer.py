import re

class Tokenizer_byt5:

    def __init__(self):
        self.pad_token = "<pad>" # id 0
        self.eos_token = "</s>"  # id 1
        self.unk_token = "<unk>" # id 2

        # vocabulary tokens, utf8 bytes, next 256 ids
        self.vocabulary_tokens = [chr(x) for x in range(256)]

        # sentinel tokens, extra_ids, next 125 ids
        self.sentinel_tokens = [f"<extra_id_{x}>" for x in range(125)]

        self.all_tokens = [self.pad_token, self.eos_token, self.unk_token, *self.vocabulary_tokens, *self.sentinel_tokens]

        # all_tokens excludes vocabulary_tokens
        self.special_tokens = list(filter(lambda token: token not in self.vocabulary_tokens, self.all_tokens))

    def text2tokens(self, text):
        delimiters  = f"({'|'.join(self.special_tokens)})"
        
        results = []
        for str in re.split(delimiters , text):
            if str in self.special_tokens:
                results.append(str)
            else:
                encoded = str.encode('utf-8')
                results.extend([chr(code) for code in bytearray(encoded)])

        if results[-1] in [self.eos_token, self.unk_token, self.pad_token]:
            pass
        else:
            results.append(self.eos_token)

        return results

    def tokens2ids(self, tokens):
        return [self.all_tokens.index(token) for token in tokens]

    def ids2tokens(self, ids):
        return [self.all_tokens[ind] for ind in ids]

    def tokens2text(self, tokens):
        array = []
        sub_array = []
        for token in tokens:
            if token in self.special_tokens:
                if len(sub_array) > 0:
                    array.append(bytearray([ord(c)  for c in sub_array]).decode("utf-8"))
                    sub_array = []
                array.append(token)
            else:
                sub_array.append(token)
        if len(sub_array) > 0:
            array.append(bytearray([ord(c)  for c in sub_array]).decode("utf-8"))       

        return ''.join(array)

    def text_clean_special_tokens(self, text):
        delimiters  = f"{'|'.join(self.special_tokens)}"
        return ''.join(re.split(delimiters , text))

    def get_config(self):
        config = {
            "pad_token": self.pad_token,
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "vocabulary_tokens": self.vocabulary_tokens,
            "sentinel_tokens": self.sentinel_tokens,
        }
        return config
       
    # def utf8string2ids(self, utf8str):
    #     offset = 3
    #     return [x + offset for x in list(utf8str.encode("utf-8"))]