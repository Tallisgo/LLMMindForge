# BPE  Tokenizer

# 什么是Tokenizer？

`Tokenizer`（分词器）将原始文本（ `raw text` ） 转化为模型能够理解的数字数列，在模型输入和输出的两个主要阶段发挥作用。

## 模型输入（Encode）

1. 分词（tokenize）：将文本拆分成词元（token），可以分为字级、词级、子词级等；

```bash
输入："我爱中国"
分词：［"我", "爱", "中国"］
```

1. token_to_id： 将每个词元映射为词汇表中唯一的ID

```bash
分词：［"我", "爱", "中国"］

id: [1101, 1231, 1145]
```

## 模型输出（Decode）

1. id_to_token: 将模型预测的序列转化为对应的token

```bash
id: [1101, 1231, 1145]

tokens：［"我", "爱", "中国"］
```

1. 输出文本： 将解码后的次元以某种规则重新拼接

```bash
tokens：［"我", "爱", "中国"］

output: "我爱中国"
```

# Transformers AutoTokenizer

```bash
pip install transformers
```

## BPE 分词器

```python
from transformers import AutoTokenizer

# 使用GPT-2的分词器 cache_dir 是指定下载的目录
tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir='./gpt2_tokenizer')

text = "Hello, how are you?"

# encode
tokens = tokenizer.tokenize(text)
print("tokens:",  tokens)  # 输出分词后的结果

token_ids = tokenizer.convert_tokens_to_ids(tokens)
print("ids:" , token_ids)

# decode
decoded_tokens = tokenizer.convert_ids_to_tokens(token_ids)
print("decode tokens: ", decoded_tokens)  # 输出解码后的结果

decoded_text = tokenizer.convert_tokens_to_string(decoded_tokens)

print("解码结果:" ,decoded_text)
```

```
tokens: ['Hello', ',', 'Ġhow', 'Ġare', 'Ġyou', '?']
ids: [15496, 11, 703, 389, 345, 30]
decode tokens:  ['Hello', ',', 'Ġhow', 'Ġare', 'Ġyou', '?']
解码结果: Hello, how are you?
```

```python
ids = tokenizer.encode(text)

print(ids)

print(tokenizer.decode(ids))
```

```
[15496, 11, 703, 389, 345, 30]
Hello, how are you?
```

# BPE 原理

`BPE` 统计语料库中相邻字符的出现频率，每次迭代目标是找到频率最高的相邻字符对并合并

$$
score_{BPE}(x,y) = freq(x,y)
$$

我们首先需要将语料库（corpus）的文本拆分成单词，假设统计结果如下：

```
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
```

### 步骤：

1. 初始化词汇表 V ，将单词拆分为字符序列，并保存到 `set` 中

```
V= {"b", "g", "h", "n", "p", "s", "u"}
```

1. 统计字符对的频次：

```python
("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)

```

1. 找到频次最高的字符对进行合并，并记录合并规则
    1. 选择频次最高的字符对，若多组字符对频次相同，任选其中一组进行合并。更新词汇表和语料库。（’u’, ‘g’）出现20次
    
    ```python
    Vocabulary: ["b", "g", "h", "n", "p", "s", "u", "ug"]
    Corpus: ("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)
    ```
    
    b. 记录合并规则
    
    ```python
    ("u", "g") -> "ug"
    ```
    
2. 重复2-3， 直到达到预定的词汇表大小

## BPE Implementation

1. 收集相关语料：

```python
corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]
```

1. 统计词频

```python
from collections import defaultdict

word_freqs = defaultdict(int)

for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

print(word_freqs)
```

1. 构建基础词汇表

```python
alphabet = []

for word in word_freqs.keys():
    for letter in word:
        if letter not in alphabet:
            alphabet.append(letter)
alphabet.sort()

print(alphabet)
```

1. 添加special token

```python
# 添加special token

vocab = ["<|endoftext|>"] + alphabet.copy()

vocab
```

1. 单词拆分

```python
# 单词拆分

splits = {word: [c for c in word] for word in word_freqs.keys()}

splits
```

1. 统计字符对

```python
def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs

pair_freqs = compute_pair_freqs(splits)
```

1. 寻找出现频率最高的字符对

```python
best_pair = ""
max_freq = None

for pair, freq in pair_freqs.items():
    if max_freq is None or max_freq < freq:
        best_pair = pair
        max_freq = freq

print(best_pair, max_freq)
```

1. *更新合并规则以及词汇表*

```python
merges = {best_pair : ''.join(best_pair)}
vocab.append(''.join(best_pair))
```

1. 更新splits

```python
def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits
```

完整流程:

```python
vocab_size = 50

while len(vocab) < vocab_size:
    pair_freqs = compute_pair_freqs(splits)
    best_pair = ""
    max_freq = None
    
    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    splits = merge_pair(*best_pair, splits)
    merges[best_pair] = best_pair[0] + best_pair[1]
    vocab.append(best_pair[0] + best_pair[1])
```

## BPE tokenization

```python
def tokenize(text):
		
		# 预分词处理： 将文本拆分成为初步的单词列表
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    
    # 将每个单词拆分为字符列表
    splits = [[l for l in word] for word in pre_tokenized_text]
    
    # 遍历所有合并规则（merges），逐步应用到拆分后的结果中
    for pair, merge in merges.items():
		    print(f"\n应用合并规则: {pair} -> {merge}")
		    
		    # 遍历每个已拆分的单词
        for idx, split in enumerate(splits):
        
			      print(f"  合并前第 {idx+1} 个单词: {split}")
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
		                # 合并字符对
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
                    
            # 更新拆分后的结果
            splits[idx] = split
		
		# 将所有拆分后的结果合并为一个 Token 列表并返回
    return sum(splits, [])
```

## 参考链接

- [https://huggingface.co/learn/llm-course/en/chapter6/5#implementing-bpe](https://huggingface.co/learn/llm-course/en/chapter6/5#implementing-bpe)