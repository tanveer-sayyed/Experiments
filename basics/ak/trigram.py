"""
    trigram with bayesian inference
    [reference]:https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part1_bigrams.ipynb
"""

from json import loads
from torch import Generator, multinomial, tensor, zeros

END = "~" # end token
START = "_" # start token
G = Generator().manual_seed(2)

counter = {}
prob = {
    "ch1" : {},
    "ch2|ch1" : {},
    "ch3|ch1,ch2":{}
    }
final_prob = prob.copy()

words = loads(open("names.txt","r").read())
words = words['payload']['blob']['rawLines']
words = [START + word + END for word in words]
unique =  sorted(set("".join(words)))
int2char = {i:c for i,c in enumerate(unique)}
char2int = {c:i for i,c in enumerate(unique)}

for word in words:
    for trigram in zip(word, word[1:], word[2:]):
        counter[trigram] = counter.get(trigram, 0) + 1
matrix = zeros(size=(len(unique),len(unique),len(unique)))
for ch1, ch2, ch3 in counter.keys():     
        matrix[char2int[ch1], char2int[ch2], char2int[ch3]] = \
            counter[(ch1, ch2, ch3)]

## prob(ch1)
for ch1 in unique: prob["ch1"][ch1] = matrix[char2int[ch1],:,:].sum().item()
prob["ch1"] = dict(sorted(prob["ch1"].items()))
temp = sum(prob["ch1"].values())
for key in prob["ch1"].keys(): prob["ch1"][key] /= temp
final_prob["ch1"] = tensor(list(prob["ch1"].values()))

# prob(ch2|ch1)
for ch2 in unique:
    for ch1 in unique:
        prob["ch2|ch1"][f"{ch2}|{ch1}"] = \
            matrix[char2int[ch1], char2int[ch2], :].sum().item()
for ch1 in unique:
    values = []
    for ch2 in unique: values.append(prob["ch2|ch1"][f"{ch2}|{ch1}"])
    final_prob["ch2|ch1"][f"*|{ch1}"] = tensor(values)
    if sum(values) != 0.0: 
        final_prob["ch2|ch1"][f"*|{ch1}"] /= tensor(values).sum()

# prob(ch3|ch1,ch2)
for ch3 in unique:
    for ch2 in unique:
        for ch1 in unique:
            prob["ch3|ch1,ch2"][f"{ch3}|{ch1},{ch2}"] = \
                matrix[char2int[ch1], char2int[ch2], char2int[ch3]].item()
for ch1 in unique:
    for ch2 in unique:
        values = []
        keys = [s for s in list(prob["ch3|ch1,ch2"].keys()) if s.endswith(f"{ch1},{ch2}")]
        for key in keys: values.append(prob["ch3|ch1,ch2"][key])
        final_prob["ch3|ch1,ch2"][f"*|{ch1},{ch2}"] = tensor(values)
        if sum(values) != 0.0:
            final_prob["ch3|ch1,ch2"][f"*|{ch1},{ch2}"] /= tensor(values).sum()

# prediction
for _ in range(5):
    out = []
    counter = 0
    while True:
        if counter == 0: # block for 1st letter
            idx = char2int[START] # always begin with the start token
            out.append(int2char[idx])
        elif counter == 1: # block for 2nd letter
            idx = multinomial(
                input=final_prob["ch2|ch1"][f"*|{out[-1]}"],
                generator=G,
                num_samples=1,
                replacement=True,
            ).item()
            out.append(int2char[idx])
        else: # after the first two letters, keep using this block ...
            idx = multinomial(
                input=final_prob["ch3|ch1,ch2"][f"*|{out[-2]},{out[-1]}"],
                generator=G,
                num_samples=1,
                replacement=True,
            ).item()
            out.append(int2char[idx])
        if idx == char2int[END]: break # ... until the end token is encountered
        counter += 1
    print("".join(out))

"""
    Output: 
        _tavi~
        _kallee~
        _lour~
        _cohnn~
        _magrah~
"""