import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.iolib.smpickle import load_pickle
from fuzzywuzzy import fuzz, process
import traceback
import googlemaps

import os

# google libraries
import polyline

# mapping and shape utils
import folium
from folium import plugins
import argparse
from util import (
    NpEncoder,
    basemaps,
    get_closest,
    colors_,
    add_legend,
    get_exit_route,
)

if __name__ == "__main__":
    description = """
    Model that provides potential evacuation routes
    """
    arg_parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    arg_parser.add_argument(
        "--config_file",
        type=str,
        help="Path to json config file. (E.g.: config.json)",
    )
    args = arg_parser.parse_args()
    print(args)
    config_file = args.config_file
    config: dict = json.load(open(config_file))
    print(config)
    googlemaps_key = config.pop("GOOGLEMAPS_KEY")
    os.environ["GOOGLEMAPS_KEY"] = googlemaps_key

    try:
        conflict_country = config.get("conflict_country", None)
        excluded_countries = config.get("excluded_countries", "")
        added_countries = config.get("added_countries", "")
        if excluded_countries =="None":
            excluded_countries=""
        if added_countries =="None":
            added_countries=""
        conflict_start = config.get("conflict_start", 2021)
        if conflict_start >2021:
            conflict_start=2021
        conflict_start=conflict_start-1
        drop_missing_data = config.get("drop_missing_data", False)
        flight_mode = config.get("flight_mode", "driving")
        number_haven_cities = config.get("number_haven_cities", 1)
        number_conflict_cities = config.get("number_conflict_cities", 15)
        percent_of_pop_leaving = config.get("percent_of_pop_leaving", 0.1)
        attraction_weight=config.get("attraction_weight",1)
        run_with_haven_cities=config.get("run_with_haven_cities",False)
        # run without haven cities to find crossings

        # read in country border data
        country_border = open("../data/country_border_data.json")
        countries_that_border = json.load(country_border)
        # get list of touching countries
        touching_list = []
        touching_list = countries_that_border[conflict_country]

        # remove any countries that are to be excluded
        indexed_list = {}
        for i, c in enumerate(touching_list):
            indexed_list[i] = c

        if len(excluded_countries) > 0:
            print(excluded_countries)
            if "," in excluded_countries:
                excluded_countries = excluded_countries.split(',')
            else:
                excluded_countries = [excluded_countries]
            for i, ex in enumerate(excluded_countries):
                country, value, ind = process.extractOne(ex, indexed_list)
                if value > 89:
                    touching_list.pop(ind)
                    indexed_list = {}
                    for i, c in enumerate(touching_list):
                        indexed_list[i] = c

        if len(added_countries) > 0:
            if ',' in added_countries:
                added_countries = added_countries.split(',')
            else:
                added_countries = [added_countries]

            if len(added_countries) > 3:
                added_countries = added_countries[0:3]
                print(f'Model run has too many added countries. We will only use: {added_countries}')
            # add any countries
            for country_v in added_countries:
                touching_list.append(country_v)

        # convert to a df
        touching_df = pd.DataFrame(touching_list, columns=["bording_countries"])
        touching_df["conflict"] = conflict_country

        # start collecting data for these countries
        # collect historic pop
        historic_pop = pd.read_csv("../data/historic_pop.csv")

        options = historic_pop["Country Name"]
        touching_df["historic_pop"] = None
        historic_pop_cols = historic_pop.columns
        indexed_col = {}
        for i, c in enumerate(historic_pop_cols):
            indexed_col[i] = c

        column, ratio_year, year_column_idx = process.extractOne(
            str(conflict_start), indexed_col
        )

        for kk, border in touching_df.iterrows():
            country, ratio, ind = process.extractOne(
                border["bording_countries"], options
            )
            touching_df.loc[kk, "historic_pop"] = historic_pop.at[ind, column]

        # get historic pop of conflict country for later use
        country, ratio, ind = process.extractOne(conflict_country, options)
        conflict_country_historic_pop = int(historic_pop.at[ind, column])

        # read in liberal democracy data
        Dem = pd.read_csv("../data/country_dem.csv")
        columnList = ["country_name", "year", "v2xeg_eqdr", "v2x_libdem"]
        country_dem = Dem[columnList]

        touching_df["v2x_libdem"] = None

        options = country_dem["country_name"].unique()

        for kk, row in touching_df.iterrows():
            country, ratio = process.extractOne(row["bording_countries"], options)
            lib = country_dem.loc[
                (country_dem["country_name"] == country)
                & (country_dem["year"] == int(conflict_start))
            ]["v2x_libdem"]

            touching_df.loc[kk, "v2x_libdem"] = lib.to_list()[0]

        # historic GDP
        historic_GDP = pd.read_csv("../data/GDP_historic.csv")
        options = historic_GDP["Country Name"]
        touching_df["historic_GDP"] = None
        historic_GDP_cols = historic_GDP.columns
        indexed_GDP_col = {}
        for i, c in enumerate(historic_GDP_cols):
            indexed_GDP_col[i] = c

        column, ratio_year, year_column_idx = process.extractOne(
            str(conflict_start), indexed_GDP_col
        )

        for kk, border in touching_df.iterrows():
            country, ratio, ind = process.extractOne(
                border["bording_countries"], options
            )
            touching_df.loc[kk, "historic_GDP"] = historic_GDP.at[ind, column]

        # normalize the GDP data
        cols_to_scale = ["historic_GDP"]
        touching_df = touching_df.rename(columns={"bording_countries": "country"})

        scaler = MinMaxScaler()
        for col in cols_to_scale:
            normed = pd.DataFrame()

            for y, x in touching_df.groupby("conflict"):
                norm_ = [
                    i[0] for i in scaler.fit_transform(x[col].values.reshape(-1, 1))
                ]
                countries = x["country"]
                conflict_ = x["conflict"]
                res = pd.DataFrame(
                    tuple(zip(countries, conflict_, norm_)),
                    columns=["country", "conflict", f"{col}_norm"],
                )
                normed = normed.append(res)
            normalized_data = pd.merge(
                touching_df,
                normed,
                left_on=["country", "conflict"],
                right_on=["country", "conflict"],
                how="right",
            )

        # modeling
        # read in model
        trained_Model = load_pickle("model/refugee_model_results.pickle")
        features_cols = [
            "historic_GDP_norm",
            "v2x_libdem",
        ]
        features_normalized = normalized_data[features_cols]

        # missing data set to 0.
        if drop_missing_data == True:
            normalized_data = normalized_data.dropna()
        else:
            normalized_data = normalized_data.fillna(0)

        features_to_predict = normalized_data[features_cols]
        shares = trained_Model.predict(features_to_predict)
        normalized_data["predicted_shares"] = shares
        border_countries_results = normalized_data[
            [
                "country",
                "conflict",
                "historic_pop",
                "historic_GDP_norm",
                "v2x_libdem",
                "predicted_shares",
            ]
        ]
        border_countries_results.to_csv(
            f"outputs/{conflict_country}_{flight_mode}_output_results.csv", index=False
        )

        # Find Largest Cities
        CITY_FILE = "../data/cities15000.txt"
        city_df = pd.read_csv(
            CITY_FILE,
            sep="\t",
            header=0,
            names=[
                "geonameid",
                "name",
                "asciiname",
                "alternatenames",
                "latitude",
                "longitude",
                "feature class",
                "feature code",
                "country code",
                "cc2",
                "admin1 code",
                "admin2 code",
                "admin3 code",
                "admin4 code",
                "population",
                "elevation",
                "dem",
                "timezone",
                "modification date",
            ],
        )
        subset_cols = ["name", "latitude", "longitude", "country code", "population"]
        city_df = city_df[subset_cols]

        # read in csv file with country names and codes.
        codes = pd.read_csv("../data/wikipedia-iso-country-codes.csv")

        # Get country codes for each have country
        options = codes["English short name lower case"]

        for kk, border in border_countries_results.iterrows():
            country, ratio, ind = process.extractOne(border["country"], options)
            border_countries_results.loc[kk, "country_code"] = codes.at[
                ind, "Alpha-2 code"
            ]

        # get conflict country country code
        country, ratio, ind = process.extractOne(
            border_countries_results["conflict"][0], options
        )
        conflict_code = codes.at[ind, "Alpha-2 code"]

        # Filter cites by country code and population
        filtered_df = city_df[city_df["country code"] == conflict_code]
        filtered_df = filtered_df.sort_values(by="population", ascending=False)
        largest_conflict_cities = filtered_df[0:number_conflict_cities]
        largest_conflict_cities["country"] = conflict_country
        largest_conflict_cities["location_type"] = "conflict_zone"

        # Do the same for camp/haven countries. These will help create more routes to find crossings.
        if run_with_haven_cities:
            largest_camp_cities = pd.DataFrame(columns=city_df.columns)
            for kk, border in border_countries_results.iterrows():
                filtered_df = city_df[city_df["country code"] == border["country_code"]]
                filtered_df["country"] = border["country"]
                filtered_df = filtered_df.sort_values(by="population", ascending=False)
                largest_camp_cities_f = filtered_df[0:number_haven_cities]
                largest_camp_cities = largest_camp_cities.append(largest_camp_cities_f)
            largest_camp_cities["location_type"] = "camp"

            # merge these two df together.
            for kk, border in largest_camp_cities.iterrows():
                largest_conflict_cities = largest_conflict_cities.append(border)

            # change column name
        locations = largest_conflict_cities.rename(columns={"name": "#name"})

        # save locations
        locations.to_csv(
            f"inputs/{conflict_country}_{flight_mode}_locations.csv", index=False
        )

        # Route Generation

        gmaps = googlemaps.Client(key=googlemaps_key)
        conflicts = locations[locations["location_type"] == "conflict_zone"]
        camps = locations[locations["location_type"] == "camp"]
        attractions = border_countries_results.copy()

        # get the crossing locations from all the routes. This is the most compute time.
        print('starting processing routes')
        crossing_locations = []
        for kk, conflict in conflicts.iterrows():
            for country in touching_list:
                try:
                    result = gmaps.directions(
                        f'{conflict["#name"]}, {conflict["country"]}',
                        country,
                        mode="driving",
                    )
                    if result:
                        for idx, i in enumerate(result[0]["legs"][0]["steps"]):
                            instr = i["html_instructions"]
                            if "Entering" in instr:
                                country_split = instr.split("Entering")[1].split("<")[0]
                                ratio = fuzz.ratio(country_split, country)
                                if ratio > 80:
                                    crossing_data = {
                                        "latitude": i["end_location"]["lat"],
                                        "longitude": i["end_location"]["lng"],
                                        "country": f"{country}",
                                    }
                                    if crossing_data not in crossing_locations:
                                        crossing_locations.append(crossing_data)
                except Exception as e:
                    print(traceback. e)
            if run_with_haven_cities:
                for kk, camp in camps.iterrows():
                    try:
                        result = gmaps.directions(
                            f'{conflict["#name"]}, {conflict["country"]}',
                            f'{camp["#name"]}, {camp["country"]}',
                            mode="driving",
                        )
                        if result:
                            for idx, i in enumerate(result[0]["legs"][0]["steps"]):
                                instr = i["html_instructions"]
                                if "Entering" in instr:
                                    country_split = instr.split("Entering")[1].split("<")[0]
                                    ratio = fuzz.ratio(country_split, camp["country"])
                                    if ratio > 80:
                                        crossing_data = {
                                            "latitude": i["end_location"]["lat"],
                                            "longitude": i["end_location"]["lng"],
                                            "country": f"{camp['country']}",
                                        }
                                        if crossing_data not in crossing_locations:
                                            crossing_locations.append(crossing_data)
                    except Exception as e:
                        traceback.print_exc()

        crossing_locations_df = pd.DataFrame(
            crossing_locations, columns=["latitude", "longitude", "country"]
        )

        # get conflict exit routes
        conflict_exit_routes = {}
        NoneType = type(None)
        for kk, conflict in conflicts.iterrows():
            closest_crossing, crossing_val = get_closest(
                conflict.latitude,
                conflict.longitude,
                crossing_locations_df,
                flight_mode,
                attraction_weight,
                attractions=attractions,
                gmaps=gmaps,
            )

            if isinstance(closest_crossing, type(None)):
                print(f'{conflict["#name"]} No routes found')
            conflict_exit_routes[conflict["#name"]] = dict(
                crossing=closest_crossing, crossing_v=crossing_val
            )
        for kk, vv in conflict_exit_routes.items():
            if not isinstance(vv["crossing"], type(None)):
                vv["crossing"] = dict(vv["crossing"])

        with open(
            f"outputs/{conflict_country}_exit_routes_{flight_mode}.json", "w"
        ) as f:
            f.write(json.dumps(conflict_exit_routes, cls=NpEncoder))

        # get directions to closes routes
        all_directions = {}
        for kk, conflict in conflicts.iterrows():
            conflict_name = conflict["#name"]
            print(f"Getting directions for conflict: {conflict_name}")

            if conflict_name in conflict_exit_routes:
                try:
                    xing = conflict_exit_routes[conflict_name]["crossing"]
                    try:
                        directions_result = gmaps.directions(
                            (conflict.latitude, conflict.longitude),
                            (xing["latitude"], xing["longitude"]),
                            mode=flight_mode,
                        )
                        directions_result[0]["name"] = xing["country"]
                        directions_result[0]["country"] = xing["country"]
                    except Exception as e:
                        print(e)
                        traceback.print_exc()
                        directions_result = None
                    all_directions[conflict_name] = directions_result
                except Exception as e:
                    print(e)
                    traceback.print_exc()
        with open(
            f"outputs/{conflict_country}_border_crossing_directions_{flight_mode}.json",
            "w",
        ) as f:
            f.write(json.dumps(all_directions))

        # read in location to convert data to correct type- this can be fixed later
        locations = pd.read_csv(
            f"inputs/{conflict_country}_{flight_mode}_locations.csv"
        )
        conflicts = locations[locations["location_type"] == "conflict_zone"]
        c_desc = conflicts.population.describe()

        def bucket_population( population):
            if population <= c_desc["25%"]:
                stroke = 2.5
            elif population <= c_desc["50%"]:
                stroke = 5
            elif population <= c_desc["75%"]:
                stroke = 7.5
            else:
                stroke = 10
            return stroke

        conflicts["stroke"] = conflicts["population"].apply(
            lambda x: bucket_population(x)
        )
        country_colors = {}
        for i, c in enumerate(touching_list):
            country_colors[c] = colors_[i]
        # plot crossings
        map = folium.Map(location=[conflicts.latitude.mean(), conflicts.longitude.mean()], zoom_start=6)

        for i, crossing in enumerate(crossing_locations):
            crossing_m = folium.Marker(
                [crossing["latitude"], crossing["longitude"]],
                popup=f'{crossing["country"]}_crossing',
                icon=folium.Icon(
                    icon="glyphicon glyphicon-road",
                    color=country_colors[crossing["country"]],
                ),
            )
            crossing_m.add_to(map)
        # Plot conflict starting points
        for kk, start in conflicts.iterrows():
            start_m = folium.Marker(
                [start.latitude, start.longitude],
                popup=start["#name"],
                icon=folium.Icon(icon="glyphicon glyphicon-fire", color="darkred"),
            )
            start_m.add_to(map)
        # plot exit routes (driving)
        if "driving" in flight_mode:
            fg_d = folium.FeatureGroup("Driving")
            for kk, vv in all_directions.items():
                stroke = int(conflicts[conflicts["#name"] == kk]["stroke"])
                population = "{:,}".format(
                    int(conflicts[conflicts["#name"] == kk]["population"])
                )
                directions = all_directions[kk]
                if not isinstance(directions, type(None)):
                    distance = directions[0]["legs"][0]["distance"]["text"]
                    duration = directions[0]["legs"][0]["duration"]["text"]
                    end_location = directions[0]["name"]

                    end_country = end_location
                    tooltip = (
                        f"Travel between <b>{kk}</b> and <b>{end_location}, {end_country}</b> by car is <b>"
                        f"{distance}</b> and takes <b>{duration}</b>.</br></br>"
                        f"<b>{population}</b> people are effected by this conflict."
                    )
                    polyline_ = polyline.decode(
                        directions[0]["overview_polyline"]["points"]
                    )
                    polyline_m = folium.PolyLine(
                        polyline_, color="#4A89F3", tooltip=tooltip, weight=stroke
                    )
                    polyline_m.add_to(fg_d)
            fg_d.add_to(map)
        if "walking" in flight_mode:
            fg_w = folium.FeatureGroup("Walking")
            for kk, vv in all_directions.items():
                stroke = int(conflicts[conflicts["#name"] == kk]["stroke"])
                population = "{:,}".format(
                    int(conflicts[conflicts["#name"] == kk]["population"])
                )
                directions = all_directions[kk]
                if not isinstance(directions, type(None)):
                    distance = directions[0]["legs"][0]["distance"]["text"]
                    duration = directions[0]["legs"][0]["duration"]["text"]
                    end_location = directions[0]["name"]

                    end_country = end_location
                    tooltip = (
                        f"Travel between <b>{kk}</b> and <b>{end_location}, {end_country}</b> by foot is <b>"
                        f"{distance}</b> and takes <b>{duration}</b>.</br></br>"
                        f"<b>{population}</b> people are effected by this conflict."
                    )
                    polyline_ = polyline.decode(
                        directions[0]["overview_polyline"]["points"]
                    )
                    polyline_m = folium.PolyLine(
                        polyline_, color="#4A89F3", tooltip=tooltip, weight=stroke
                    )
                    polyline_m.add_to(fg_w)
            fg_w.add_to(map)


        # plot exit routes (transit)
        if "transit" in flight_mode and run_with_haven_cities:
            fg_t = folium.FeatureGroup("Transit")
            for kk, vv in all_directions.items():
                stroke = int(conflicts[conflicts["#name"] == kk]["stroke"])
                population = "{:,}".format(
                    int(conflicts[conflicts["#name"] == kk]["population"])
                )
                directions = all_directions[kk]
                if not isinstance(directions, type(None)):
                    if len(directions) > 0:
                        distance = directions[0]["legs"][0]["distance"]["text"]
                        duration = directions[0]["legs"][0]["duration"]["text"]
                        end_location = directions[0]["name"]
                        end_country = camps[
                            camps["#name"] == end_location
                        ].country.values[0]
                        tooltip = (
                            f"Travel between <b>{kk}</b> and <b>{end_location}, {end_country}</b> by transit is <b>"
                            f"{distance}</b> and takes <b>{duration}</b>.</br></br>"
                            f"<b>{population}</b> people are effected by this conflict."
                        )
                        polyline_ = polyline.decode(
                            directions[0]["overview_polyline"]["points"]
                        )
                        polyline_m = folium.PolyLine(
                            polyline_, color="#7570b3", tooltip=tooltip, weight=stroke
                        )
                        polyline_m.add_to(fg_t)

            fg_t.add_to(map)

        basemaps["Google Satellite Hybrid"].add_to(map)
        # basemaps['Esri Satellite'].add_to(map)
        # basemaps['Google Satellite'].add_to(map)
        basemaps["Google Maps"].add_to(map)

        # Add a layer control panel to the map.
        # map.add_child(folium.LayerControl())
        plugins.Fullscreen().add_to(map)

        map = add_legend(map)
        # save map
        map.save(f"maps/Map.html")

        # Calculate Recipient Country Refugee Counts
        conflicts = locations[locations["location_type"] == "conflict_zone"]
        conflicts = conflicts.apply(
            lambda row: get_exit_route(row, flight_mode, conflict_exit_routes), axis=1
        )

        border_countries = border_countries_results.copy()

        conflict_country_historic_pop = int(conflict_country_historic_pop)
        conflicts["pop_percent_of_conflict_cities"] = (
            conflicts["population"] / conflicts["population"].sum()
        )
        conflicts[f"refugee_estimated_leaving_via_{flight_mode}"] = conflicts[
            "pop_percent_of_conflict_cities"
        ] * (conflict_country_historic_pop * percent_of_pop_leaving)
        conflicts['conflict_year'] = conflict_start

        # reduce size of output file
        COL = ["#name", "country", "conflict_year", f"{flight_mode}_destination", "latitude", "longitude",
               f"refugee_estimated_leaving_via_{flight_mode}"]
        reduced_conflicts = conflicts[COL]
        reduced_conflicts = reduced_conflicts.rename(columns={"#name": "origin city", "country": "origin country",
                                                              f"{flight_mode}_destination": "destination country",
                                                              f"refugee_estimated_leaving_via_{flight_mode}": "total refugees"})
        # save df
        reduced_conflicts['total refugees'] = reduced_conflicts['total refugees'].astype('int')
        reduced_conflicts.to_csv(f'outputs/{conflict_country}_{flight_mode}_total_refugees.csv', index=False)

        country_level_refugee = pd.DataFrame(
            data=reduced_conflicts.groupby(['destination country'])["total refugees"].sum())

        country_level_refugee.reset_index(inplace=True)
        country_level_refugee = country_level_refugee.rename(columns={'destination country': 'country'})

        country_level_refugee.to_csv(f'outputs/{conflict_country}_{flight_mode}_total_refugees_by_country.csv', index=True)


    except Exception as e:
        traceback.print_exc()
