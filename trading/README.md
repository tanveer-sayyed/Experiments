## The Background
In Stock market, its machines talking to machines and competing with machines. And machines are not humans! They "think" diffirently; its certain that only a machine can defeat a machine, and not a human. Hence it is important that the algorithms that we create have zero human intervention, ideally. Following is an attempt to build the same, for intra-day frequency. The attempt here is to be able to take an opposite position, i.e. if a market is falling, strike on a call; if the market is rising strike on a put.

## The Experiemnt
The following experiment is an attempt to predict if the direction of the next day for NIFTY Bank index. Backtesting begins from 1st Aug, 2021.

	date	    prediction	close	   ground_truth
2021-07-30		         34584.35	
2021-08-02	 HIGHER	   34710.0	  HIGHER
2021-08-03	 HIGHER	   35207.45	  HIGHER
2021-08-04	 LOWER	   36028.05	  HIGHER
2021-08-05	 LOWER	   35834.75	  LOWER
2021-08-06	 LOWER	   35809.25	  LOWER
2021-08-09	 LOWER	   36028.95	  HIGHER
2021-08-10	 HIGHER	   36034.1	  HIGHER
2021-08-11	 HIGHER	   35806.4	  LOWER
2021-08-12	 LOWER	   35937.05	  HIGHER
2021-08-13	 LOWER	   36169.35	  HIGHER
2021-08-16	 HIGHER	   36094.5	  LOWER
2021-08-17	 LOWER	   35867.45	  LOWER
2021-08-18	 HIGHER	   35554.5	  LOWER
2021-08-20	 LOWER	   35033.85	  LOWER
2021-08-23	 HIGHER	   35124.4	  HIGHER
2021-08-24	 HIGHER	   35712.1	  HIGHER
2021-08-25	 LOWER	   35586.25	  LOWER
2021-08-26	 HIGHER	   35617.55	  HIGHER
2021-08-27	 LOWER	   35627.8	  HIGHER
2021-08-30	 HIGHER	   36347.65	  HIGHER
2021-08-31	 HIGHER	   36424.6	  HIGHER
2021-09-01	 HIGHER	   36574.3	  HIGHER
2021-09-02	 LOWER	   36831.3	  HIGHER


As can be seen in the above image out of the total business days in the month of August-2021 (which includes a holiday), the model was able to predict correctly the direction for 13 days.

Aug, 2021 :: 61.9047% [13/21]
Sep, 2021 :: 50.0000% [1/2]

The overall current accuracy of the model is thus: 14/23 = 60.8695%

## Future
Predict also the best price at which one can hold positions for intra-day trade. The work is still in progress.
