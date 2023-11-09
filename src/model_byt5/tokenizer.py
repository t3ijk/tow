import re

# 0-2 tokens
pad_token = "<pad>" # 0
eos_token = "</s>" # 1
unk_token = "<unk>" # 3

# vocabulary tokens, utf8 bytes, next 256
vocabulary_tokens = [chr(x) for x in range(256)]

# sentinel tokens, extra_ids, next 125

# sentinel_tokens = [f"<extra_id_{x}>" for x in range(125)]
sentinel_tokens = [
        "<extra_id_0>",
        "<extra_id_1>",
        "<extra_id_2>",
        "<extra_id_3>",
        "<extra_id_4>",
        "<extra_id_5>",
        "<extra_id_6>",
        "<extra_id_7>",
        "<extra_id_8>",
        "<extra_id_9>",
        "<extra_id_10>",
        "<extra_id_11>",
        "<extra_id_12>",
        "<extra_id_13>",
        "<extra_id_14>",
        "<extra_id_15>",
        "<extra_id_16>",
        "<extra_id_17>",
        "<extra_id_18>",
        "<extra_id_19>",
        "<extra_id_20>",
        "<extra_id_21>",
        "<extra_id_22>",
        "<extra_id_23>",
        "<extra_id_24>",
        "<extra_id_25>",
        "<extra_id_26>",
        "<extra_id_27>",
        "<extra_id_28>",
        "<extra_id_29>",
        "<extra_id_30>",
        "<extra_id_31>",
        "<extra_id_32>",
        "<extra_id_33>",
        "<extra_id_34>",
        "<extra_id_35>",
        "<extra_id_36>",
        "<extra_id_37>",
        "<extra_id_38>",
        "<extra_id_39>",
        "<extra_id_40>",
        "<extra_id_41>",
        "<extra_id_42>",
        "<extra_id_43>",
        "<extra_id_44>",
        "<extra_id_45>",
        "<extra_id_46>",
        "<extra_id_47>",
        "<extra_id_48>",
        "<extra_id_49>",
        "<extra_id_50>",
        "<extra_id_51>",
        "<extra_id_52>",
        "<extra_id_53>",
        "<extra_id_54>",
        "<extra_id_55>",
        "<extra_id_56>",
        "<extra_id_57>",
        "<extra_id_58>",
        "<extra_id_59>",
        "<extra_id_60>",
        "<extra_id_61>",
        "<extra_id_62>",
        "<extra_id_63>",
        "<extra_id_64>",
        "<extra_id_65>",
        "<extra_id_66>",
        "<extra_id_67>",
        "<extra_id_68>",
        "<extra_id_69>",
        "<extra_id_70>",
        "<extra_id_71>",
        "<extra_id_72>",
        "<extra_id_73>",
        "<extra_id_74>",
        "<extra_id_75>",
        "<extra_id_76>",
        "<extra_id_77>",
        "<extra_id_78>",
        "<extra_id_79>",
        "<extra_id_80>",
        "<extra_id_81>",
        "<extra_id_82>",
        "<extra_id_83>",
        "<extra_id_84>",
        "<extra_id_85>",
        "<extra_id_86>",
        "<extra_id_87>",
        "<extra_id_88>",
        "<extra_id_89>",
        "<extra_id_90>",
        "<extra_id_91>",
        "<extra_id_92>",
        "<extra_id_93>",
        "<extra_id_94>",
        "<extra_id_95>",
        "<extra_id_96>",
        "<extra_id_97>",
        "<extra_id_98>",
        "<extra_id_99>",
        "<extra_id_100>",
        "<extra_id_101>",
        "<extra_id_102>",
        "<extra_id_103>",
        "<extra_id_104>",
        "<extra_id_105>",
        "<extra_id_106>",
        "<extra_id_107>",
        "<extra_id_108>",
        "<extra_id_109>",
        "<extra_id_110>",
        "<extra_id_111>",
        "<extra_id_112>",
        "<extra_id_113>",
        "<extra_id_114>",
        "<extra_id_115>",
        "<extra_id_116>",
        "<extra_id_117>",
        "<extra_id_118>",
        "<extra_id_119>",
        "<extra_id_120>",
        "<extra_id_121>",
        "<extra_id_122>",
        "<extra_id_123>",
        "<extra_id_124>"
    ]


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
