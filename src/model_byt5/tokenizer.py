import re

class Tokenizer_byt5:

    def __init__(self):
        self.pad_token = "<pad>" # id 0
        self.eos_token = "</s>"  # id 1
        self.unk_token = "<unk>" # id 2

        # vocabulary tokens, utf8 bytes, next 256 ids
        self.vocabulary_tokens = [chr(x) for x in range(256)]

        # extra_ids, next 125 ids
        # According "Rather than adding 100 new tokens for the sentinels, we find it sufficient to reuse the final 100 byte IDs". Byt5 paper https://aclanthology.org/2022.tacl-1.17.pdf
        # So 'extra_ids' not used as sentinel tokens, and in pre-training 100 sentinels from [258] to [159], values not in range (U+0000, U+007F), so must be safe. 
        self.extra_tokens = [f"<extra_id_{x}>" for x in range(125)]

        self.all_tokens = [self.pad_token, self.eos_token, self.unk_token, *self.vocabulary_tokens, *self.extra_tokens]

        # all_tokens excludes vocabulary_tokens
        self.special_tokens = list(filter(lambda token: token not in self.vocabulary_tokens, self.all_tokens))

    def text2ids(self, text):
        return self.tokens2ids(self.text2tokens(text))  

    def text2tokens(self, text):
        delimiters  = f"({'|'.join(self.special_tokens)})"
        
        results = []
        for str in re.split(delimiters , text):
            if str in self.special_tokens:
                results.append(str)
            else:
                encoded = str.encode('utf-8')
                results.extend([chr(code) for code in bytearray(encoded)])

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
                    array.append(bytearray([ord(c)  for c in sub_array]).decode("utf-8", errors='ignore'))
                    sub_array = []
                array.append(token)
            else:
                sub_array.append(token)
        if len(sub_array) > 0:
            array.append(bytearray([ord(c)  for c in sub_array]).decode("utf-8", errors='ignore'))       

        return ''.join(array)
    
    def ids2text(self, ids):
        return self.tokens2text(self.ids2tokens(ids))

    def text_clean_special_tokens(self, text):
        delimiters  = f"{'|'.join(self.special_tokens)}"
        return ''.join(re.split(delimiters , text))

    def get_config(self):
        config = [
            {"pad_token": self.pad_token},
            {"eos_token": self.eos_token},
            {"unk_token": self.unk_token},
            {"vocabulary_tokens": self.vocabulary_tokens},
            {"extra_tokens": self.extra_tokens},
        ]
        return config
    
    def __call__(self, text, max_length):

        ids = self.text2ids(text)
        ids = ids[:max_length-1]
        ids.append(1)
        return ids
       