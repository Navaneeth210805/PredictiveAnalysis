import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Set seed for reproducibility
np.random.seed(0)

# Generate synthetic classification data
X, y = make_classification(
    n_samples=1000, 
    n_features=4, 
    n_classes=2,
    n_informative=3, 
    n_redundant=1, 
    flip_y=0.1, 
    class_sep=1.5, 
    random_state=42
)

df = pd.DataFrame(X, columns=['Temperature', 'Run_Time', 'Pressure', 'Humidity'])

df['Downtime_Flag'] = y
df['Machine_ID'] = np.arange(1, len(df) + 1)

df['Temperature'] = (df['Temperature'] * 20 + 60).clip(20, 100)
df['Run_Time'] = (df['Run_Time'] * 25 + 50).clip(0, 100)
df['Pressure'] = (df['Pressure'] * 25 + 50).clip(0, 100)
df['Humidity'] = (df['Humidity'] * 25 + 50).clip(0, 100) 

df.to_csv('data.csv', index=False)
print(df.head())
