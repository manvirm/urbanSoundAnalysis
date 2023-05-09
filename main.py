import pandas as pd
import numpy as np
import librosa
import librosa.display
import random
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('Urban Sound Dataset.csv')
df.load()

# Exploratory Data Analysis
data, sampling_rate = librosa.load('rain/1.way')
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data, sr=sampling_rate)

index = random.choice(df.index)

print('Class: ', df['Class'][index])
data, sampling_rate = librosa.load('Train/' + str(df['ID'][index] + '.wav'))

plt.figure(figsize=(12,4))
librosa.display.waveplot(data, sr=sampling_rate)

