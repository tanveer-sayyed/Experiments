from json import loads
from torch import int32, zeros

END = "~" # end token
START = "_" # start token

words = loads(open("names.txt","r").read())
words = words['payload']['blob']['rawLines']
words = [START + word + END for word in words]
unique =  sorted(set("".join(words)))
int2char = {i:c for i,c in enumerate(unique)}
char2int = {c:i for i,c in enumerate(unique)}

counter = {}
for word in words:
    for trigram in zip(word, word[1:], word[2:]):
        counter[trigram] = counter.get(trigram, 0) + 1
# counter = {k:v for k,v in sorted(counter.items(), key=lambda x: x[0][2])}

matrix = zeros(size=(len(unique),len(unique),len(unique)), dtype=int32)
for ch1, ch2, ch3 in counter.keys(): 
    matrix[
        char2int[ch1],
        char2int[ch2],
        char2int[ch3],
        ] = counter[(ch1, ch2, ch3)]

matrix = matrix.float()
for char in unique:
    matrix[:,:,char2int[char]] /= matrix[:,:,char2int[char]].sum()

