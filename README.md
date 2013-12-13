################################################################################
## Description
################################################################################

Our code and process notebook for our analysis and predictive modeling
approaches to understand directional stock movements. We have launched a website
to showcase our work at https://sites.google.com/site/predictingstockmovement/

################################################################################
## Objective
################################################################################
The objective was the predict the directional movement of a stock on day 10, given
the opening, closing, min, max, and volume of a stock in the previous 9 days 
(and given the opening price of a stock on day 10)

################################################################################
## Team
################################################################################
Team Name: Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo

Team Members: William Chen '14, Sebastian Chiu '14, Salena Cui '15, Carl Gao '15

################################################################################
## Result
################################################################################
We submitted our Ridge-Random Forest model to the Boston Data Week hackathon
hosted at Hack/Reduce. Information about the competition is available at 
https://inclass.kaggle.com/c/boston-data-festival-hackathon

We placed 1st out of 21 teams, and were able to achieve a 94.119% AUC on the
private leaderboard

################################################################################
## Files
################################################################################
process.ipynb notebook
	Notebook describing our work and our main contributions

model_tuner.py or model_tuner.ipynb
	Find the parameters for the ridge regression and random forest regression
	that we used

model_stacker.py or model_stacker.ipynb
	Stack our two final models

test.csv and training.csv
	Official data

predictions/
	contains both final submissions. Winning submission is in the stacker directory

################################################################################
## Official Data Description
################################################################################

training.csv - time series for 94 stocks (94 rows). First number in each row is the stock ID. Then data for 500 days. Data for each day contain - day opening price, day maximum price, day minimum price, day closing price, trading volume for the day. Price data normalised to the first day opening price.

test.csv - data to create prediction. Data provided for 25 time segments. Each segment contains data for the same 94 stocks. Each segment has opening, max, min, closing, volume data for 9 days and opening for day #10. Each line of the file starts with segment number following by stock ID and then price and volume data organized by day the same way as training set.  Price data normalised to the first day opening price.

Each line in train.csv and test.csv contains consecutive trading days. Days when market was closed were excluded. Thus day N may be Friday and day N+1 may be Monday or even Tuesday if Monday was a holiday. 

Value to predict - probability of stock moving up from  opening of day 10 to closing of day 10. Prediction should be in 0-1 range, where 1 - "stock surely will go up", 0- "stock surely will go down".

Test set is randomly sampled without overlapping from year following training data time period.