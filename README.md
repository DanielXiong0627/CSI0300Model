To use this model:

1. Run the preprocessing file. The model was tested on the HS300 dataset in the folder but should work for any dataset with the same format.

2. Run the main file with a CSV containing price data and technical indicators. It will:

Train the model using specified time windows
Generate two types of signals (regression and classification)
Output three CSV files: signal.csv, dates.csv, and labels.csv.
Place all of these files into the stategy_testing folder

3. Go into the strategy_testing folder, and run strategy.py. This will:

Take the generated signals as input
Calculate position weights based on z-scores
Output weights.csv containing the final trading positions (-1 to +1)

Use these weights for trading by:

Taking long positions when weight > 0 (stronger buy signals = larger positions)
Taking short positions when weight < 0 (stronger sell signals = larger short positions)
Staying neutral when weight = 0

4. You may also test the model by opening backtest.py, which will provide testing based on the time interval not used in the training or testing of the model.



Note that the model requires considerable historical data for training and should be periodically retrained as new data becomes available.
