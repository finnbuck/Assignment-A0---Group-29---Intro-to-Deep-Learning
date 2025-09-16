import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler # Preprocessing recommended by: https://www.geeksforgeeks.org/data-analysis/principal-component-analysis-with-python/

train_in = pd.read_csv('train_in.csv')
train_out = pd.read_csv('train_out.csv')