DESCRIPTION
This model was created for the Boston Data Week hackathon hosted at Hack/Reduce.

OBJECTIVE
The objective was the predict the directional movement of a stock on day 10, given
the opening, closing, min, max, and volume of a stock in the previous 9 days 
(and given the opening price of a stock on day 10)

TEAM
Team Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo
William Chen
Sebastian Chiu
Salena Cui
Carl Gao

FILES

model_tuner.py or model_tuner.ipynb
Find the parameters for the ridge regression and random forest regression
that we used

model_stacker.py or model_stacker.ipynb
Stack our two final models

test.csv and training.csv
Official data

predictions/
contains both final submissions. Winning submission is in the stacker directory