 
In Stock market, its machines talking to machines and competing with machines. And machines are not humans! They "think" diffirently; its certain that only a machine can defeat a machine, and not a human. Hence it is important that the algorithms that we create have zero human intervention, ideally. Following is an attempt to build the same, for intra-day frequency. The attempt here is to be able to take an opposite position, i.e. if a market is falling, strike on a call; if the market is rising strike on a put.

The following image shows if market is exptected to rise or fall. Also, the best tradable intra-day value, i.e. which should yield the "minimum" return, is the output of the algorithm.

Image-1

![image](https://user-images.githubusercontent.com/45352897/127315166-6a8bf69c-4c0b-4c8c-8917-34b8a8143fc8.png)


Above it can be seen that the day's prediction is normalised to account for short-term variations. The adjustment is decided by the algorithm and not humans.
This adjusted value is thus more robust. Even medium-term variations would be considered, which is still a work in progress.

The below graph shows the testing result of above model, which is, the prediction of the best target value for the day. Note that the the same value also serves another purpose - predicting the direction. As you can see the direction is correct in more than 90% of the cases!

Image-2
![image](https://user-images.githubusercontent.com/45352897/127315268-eac0b1ed-d7ca-495d-96fa-3b76fa6e57a6.png)

Now Image-1 shows that the market is expected to fall. So a put option would be strategy of many. But what about taking an opposite position? What if we purchase a
call instead?! Let's see how our best target(34516.24) fares on the charts: (see the solid blue line)

Image-3
![image](https://user-images.githubusercontent.com/45352897/127315895-f4f2c89f-e1ca-4472-9d54-378d57318187.png)

Furture work:
Based on the slope, the algorithm decides to buy only calls or only puts or both calls and puts (along with their thresholds, calculated by the model).
