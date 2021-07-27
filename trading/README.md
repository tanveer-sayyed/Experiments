 
In Stock market, its machines talking to machines and competing with machines. And machines are not humans! They "think" diffirently; its certain that only a machine can defeat a machine, and not a human. Hence it is important that the algorithms that we create have zero human intervention, ideally. Following is an attempt to build the same, for intra-day frequency.

The best tradable intra-day value, i.e. which should yield the "minimum" return, is the output of the algorithm.

![image](https://user-images.githubusercontent.com/45352897/127092736-73bf7eae-2927-45a4-a726-04074b5ee443.png)

Above it can be seen that the day's prediction is normalised to account for short-term variations. The adjustment is decided by the algorithm and not humans.
This adjusted value is thus more robust. Even medium-term variations would be considered, which is still a work in progress.

The below graph shows the testing result of model. As you can see the direction is correct in more than 90% of the cases!
Also the distance between the green and the red dot is the profit maximiser/loss minimiser buffer; the more the distance the more the scope for wealth preservation(contingent upon the direction).

![2021-07-26 12:57:23 712827_Close_ FUTURE](https://user-images.githubusercontent.com/45352897/127092812-1c26f1a7-3aa4-484c-a3b6-7c41d69fc2f6.png)

Upon prediction the next step is to maximise the return by activating a momentum strategy, after a particular threshold. This threshold is decided by the model.

Furture work:
Based on the slope the algorithm decides to buy only calls or only puts or both calls and puts (along with their thresholds, calculated by the model).
