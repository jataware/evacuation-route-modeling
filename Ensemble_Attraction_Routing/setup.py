import glob
import os
import pandas as pd
data=pd.read_csv('../data/GDP_preprocessed.csv', skiprows=3)
print(data.head())
data.to_csv("../data/GDP_historic.csv", index=False)

pop_data=pd.read_csv('../data/POP_preprocessed.csv', skiprows=3)
print(pop_data.head())
pop_data.to_csv("../data/historic_pop.csv", index=False)

vdem_data=pd.read_csv('../data/vdem_preprocessed.csv')
columnList=["country_name","year","v2xeg_eqdr","v2x_libdem"]
country_dem=vdem_data[columnList]
country_dem.to_csv("../data/country_dem.csv")
print('Done saving new data')
