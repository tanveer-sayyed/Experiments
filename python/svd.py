import torch
from scipy.linalg import svd
my_party = torch.zeros(24)
my_party[18] = 1 # a two hour party from 1800 to 2000 hrs
my_party[19] = 1
print(f"{my_party=}")

friends_schedule = torch.zeros((3, 24))
friends_schedule[0, 17] = 1   # first friend is available at 1700
friends_schedule[1, 19] = 0.5 # second one, for half an hour / half chances to attend the party
friends_schedule[2, 18] = 1   # third friend is available at 1800 hours
print(f"{friends_schedule=}")

probs = my_party @ friends_schedule.T
print("so who will be attending the party:")
print(f"first friend chances: {probs[0]}")
print(f"second friend chances: {probs[1]}")
print(f"third friend chances: {probs[2]}")

U, s, VT = svd(friends_schedule.data.numpy())
