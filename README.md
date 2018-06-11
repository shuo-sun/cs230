# CS230 Project: Air Quality Forecasting Using Convolutional LSTM

*Authors: Shuo Sun and Gaoxiang Liu*

## Requirements

This code requires Python3 with Pytorch 4.0 and NumPy.

## Task

Using past 24 hour weather and air quality data to predict future 48-hour air quality metrics of 35 air quality stations in Beijing.

Details of the task: https://biendata.com/competition/kdd_2018/

## Usage

1. Download all Beijing data from KDD competition 2018. https://biendata.com/competition/kdd_2018/data/, and put all files in `data` directory

2. To train the model with default config. Just run the main.py file. The model will be saved to `models/model_<loss>.md` for each epoch.

3. To customize model configeration, try using Jupyter notebook and follow the procedure in the main function.
