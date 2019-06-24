import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

names = []
filenames = []

info = [("Hidden Layer Experiment Trial 1", "Hidden Layer Experiment 2.csv"), ("Hidden Layer Experiment Trial 2", "Hidden Layer Experiment 2 2.csv"), ("Hidden Layer Experiment Trial 3", "Hidden Layer Experiment 2 3.csv")]
IN_DIRECTORY = "./Data/"
OUT_DIRECTORY = "./Figures/"

for name, filename in info:
    df = pd.read_csv(IN_DIRECTORY + filename, index_col=0)
    df_rolling = df.rolling(50).mean()

    df_rolling.plot()
    plt.xlabel("Episode")
    plt.ylabel("Rolling Mean Reward (50)")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(OUT_DIRECTORY + "Rolling " + name)

