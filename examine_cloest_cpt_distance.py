import pandas as pd
import numpy as np

df = pd.read_csv("/home/arr65/vs30_data_output/closest_cpt_distance.csv")

counts_closest = df['closest_cpt_name'].value_counts()

print()

cpts_never_thought_to_be_closest = df.loc[~df['cpt_name'].isin(df['closest_cpt_name']), 'cpt_name']

print()
