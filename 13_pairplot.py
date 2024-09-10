import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../dataset/ch_04/feature_soilnutrients_VI.csv')
unique_values = df['feature_soil nutrients'].unique()
print(unique_values)

sns.pairplot(df, hue="feature_soil nutrients", 
             vars=["NDVI", "GNDVI", "NDRE", "OSAVI", "BLUE", "GREEN", "RED", "REDEDGE", "NIR"], 
             markers=["o", "s", "D", "^"])
plt.show()
