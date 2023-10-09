from json import loads
from torch import float32, Generator, multinomial, tensor, zeros

END = "~" # end token
START = "_" # start token
G = Generator().manual_seed(103)

prob = { # used for saving probability distributions
    "ch1" : {},
    "ch2|ch1" : {},
    "ch3|ch1,ch2":{},
    }
counter = {}
words = loads(open("names.txt","r").read())
words = words['payload']['blob']['rawLines']
words = [START + word + END for word in words]
unique =  sorted(set("".join(words)))
int2char = {i:c for i,c in enumerate(unique)}
char2int = {c:i for i,c in enumerate(unique)}
for word in words:
    for trigram in zip(word, word[1:], word[2:]):
        counter[trigram] = counter.get(trigram, 0) + 1
matrix = zeros(size=(len(unique),len(unique),len(unique)), dtype=float32)
for ch1, ch2, ch3 in counter.keys():     
        matrix[char2int[ch1], char2int[ch2], char2int[ch3]] = \
            counter[(ch1, ch2, ch3)]

## prob(ch1) :: O(n)
for ch1 in unique: prob["ch1"][ch1] = matrix[char2int[ch1],:,:].sum().item() # count
prob["ch1"] = dict(sorted(prob["ch1"].items()))
temp = sum(prob["ch1"].values())
for key in prob["ch1"].keys(): prob["ch1"][key] /= temp # normalise
prob["ch1"] = tensor(list(prob["ch1"].values()))

# prob(ch2|ch1) :: O(n**2)
for ch1 in unique:
    values = []
    for ch2 in unique: values.append(
            matrix[char2int[ch1], char2int[ch2], :].sum().item() # count
            )
    prob["ch2|ch1"][f"*|{ch1}"] = tensor(values)
    if sum(values) != 0.0: 
        prob["ch2|ch1"][f"*|{ch1}"] /= tensor(values).sum() # normalise

# prob(ch3|ch1,ch2) :: O(n**3) <-- the CURSE of dimenstionality begins; the curse creates sparse matrices
for ch1 in unique:
    for ch2 in unique:
        values = []
        for ch3 in unique:
            values.append(
                    matrix[char2int[ch1], char2int[ch2], char2int[ch3]].item() # count
                    )
        prob["ch3|ch1,ch2"][f"*|{ch1},{ch2}"] = tensor(values)
        if sum(values) != 0.0:
            prob["ch3|ch1,ch2"][f"*|{ch1},{ch2}"] /= tensor(values).sum() # normalise

# prediction
for _ in range(5):
    out = []
    counter = 0
    while True:
        if counter == 0:
            idx = char2int[START]      # always begin with the start token
            out.append(int2char[idx])
        elif counter == 1:             # 2nd letter is sampled by feeding in the 
            idx = multinomial(         # probability distribution from the first letter
                input=prob["ch2|ch1"][f"*|{out[-1]}"],
                generator=G,
                num_samples=1,
                replacement=True,
            ).item()
            out.append(int2char[idx])
        else:                          # subsequent letters are sampled by feeding
            idx = multinomial(         # in previous probability distributions  ...
                input=prob["ch3|ch1,ch2"][f"*|{out[-2]},{out[-1]}"],
                generator=G,
                num_samples=1,
                replacement=True,
            ).item()
            out.append(int2char[idx])
        if idx == char2int[END]: break # ... until the end token is encountered
        counter += 1
    print("".join(out))
    """
        raw output: 
            _yoh~
            _kiya~
            _johieane~
            _jiahara~
            _ivikam~
    """
