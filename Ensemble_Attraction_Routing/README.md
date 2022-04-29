# Migration Route Modeling
This is a generalized model that predicts refugee routes from a conflict country to "haven" countries. 


## The model
This model calculates an attraction score for each haven country based on historic GDP and liberal democracy index. The attraction scores are combined with a distance minimization heuristic to find the most "attractive" route a refugee would take. 
We implement this model by first running the model over each bordering country of a conflict county to get the attraction scores. Then we use google maps api to get directions from the 20 most populous conflict country cities to each haven country. If directions are not found to that country we try to find directions to the top 3 most populous cities in the haven country. Once we get a valid direction we parse the directions to find the border crossing and calculate the total duration and total distance from the conflict city to the border crossing. So each conflict country will have directions to each haven country and the directions, duration, and distance will be stopped at each associated border crossing. 
With those duration values we determine which of the haven countries each conflict city will go to. Each duration value is feed into a formula that takes attraction score and attraction weight into account to either increase or decrease the duration of the trip. We allow an adjustable weighting called `attraction_weight` which is between `0 and 1` where `0` means the decision is purely based on duration of trip and `1` means it is heavily influenced on attraction score of the country. For each conflict city, after we take attraction score and attraction weight and modify the duration of the trip for each border crossing, we select the shortest duration. 



### Data
We gathered historic data on democratic conditions within each haven country one year prior to the conflict from V-Dem, including their liberal democracy index (v2x_libdem). We collect historic GDP from World Bank. 
The model is a simple linear regression uses these two features normalized by all the haven countries. The output is the attraction score. 

## Running the model
To run the model you will need a google api key. Once you have your google api key follow these steps:
1. Clone this repo.
```
 git clone https://github.com/jataware/migration-route-modeling.git
```
2. navigate to the  Ensemble_Attraction_Routing folder
```
cd Ensemble_Attraction_Routing
```
3. run the setup command. This adds folders to the directory
```
./setup.sh
```
4. Now install the requirements.
```
pip install -r requirements.txt
```
5. Update the config file for the model run. Edit config.json.ensemble. This is where you will add your google api key. **conflict_country** is where the conflict will occur. **flight_mode** is the type of transportation the currently supported choices are "driving" or "walking". **conflict_start** is the year the conflict starts. We take this year and collect the appropriate GDP, historic liberal democracy index and historic population based on that year. *Note any conflict_year after 2020 will use 2020 data since that is the latest data from World Bank. **number_haven_cities** does not need to be updated. This just collect the top 4 most populous haven cities in case directions are not found to the haven country.  **number_conflict_cities** can be changed but its best to use 10 to 20 most populous cities in the conflict country. **drop_missing_data** is if you want to drop countries that are missing data. Default is to fill in the features with 0 values, but dropping them might make sense in certain situations. **added_countries** are any countries you would like to add to the analysis. These do not have to be bordering the conflict country. These should be a string value separated by commas. Max number of added countries is 3. **excluded_countries** are any countries you would like to exclude from the analysis. These should be comma separated. There is no max for excluding countries. **percent_of_pop_leaving** This sets the total percent of the conflict country population to become refugees. Default is 10% of the population. If conflict year is not 2020 or greater it will use historic population for that year. **attraction_weight** is how much weight to put on the attraction scores for each country in the decision process. re `0` means the decision is purely based on duration of trip and `1` means it is heavily influenced on attraction score of the country.
Here is an example. 
```
{
"GOOGLEMAPS_KEY":"Your Key Here",  
"conflict_country":"Ukraine",  
"flight_mode":"driving",  
"conflict_start":2023,  
"number_haven_cities":4,  
"number_conflict_cities":10,  
"drop_missing_data":false,  
"added_countries":"None",  
"excluded_countries":"Russia,Belarus",  
"percent_of_pop_leaving": 0.1,  
"attraction_weight":0.3
}
```

## Outputs
There are a few output files from a model run. These will be found in the outputs/ folder.
The first one is {conflict_country}_{flight_mode}_output_results.csv. In my example run it would be Ukraine_Driving_output_results.csv. This file has each country's GDP, Liberal Democracy, historic population and Attraction Score (predicted_shares).  Next is the {conflict_country}_{flight_mode}_total_refugee.csv file which has each conflict city's predicted number of refugees, lat and long of border crossing and the associated destination country. Lastly, there is {conflict_country}_{flight_mode}_total_refugee_by_country.csv which has each haven country and the predicted number of refugees.
All json files that are outputed are data on directions and duration times.
### Maps
Each model run has an output map. This map should plot each conflict city, each border crossing found, and the route chosen from each conflict city given the conditions. For the example it looks like this. 
