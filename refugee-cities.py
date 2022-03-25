import argparse
from enum import Enum
import json
import os

import geopandas as gpd
import pandas as pd
import math
import numpy as np
import shapely
from shapely.geometry import Point
import googlemaps
import pgeocode
import pyproj
import polyline

class TravelModes(Enum):
    Driving = "DRIVING"
    Walking = "WALKING"

CITY_FILE = "cities1000.txt"


def get_directions(start, end):
    '''
    This function takes in a start and end location from `mlocations.csv`
    and obtains the Google Directions for them.
    '''
    directions_result = googlemaps.directions(
        (start.latitude, start.longitude),
        (end.latitude, end.longitude),
        mode="driving")
    return directions_result


def find_cities(
        start_lat,
        start_lon,
        disaster_radius_km,
        flight_radius_km,
        travel_mode="DRIVING",
        extra_filters=[],
):
    AEQD_STR = pyproj.Proj(f"+proj=aeqd +units=km +lat_0={start_lat} +lon_0={start_lon}")
    EPSG_STR = "EPSG:4326"

    googlemaps_key = os.environ.get("GOOGLEMAPS_KEY")
    gmaps = googlemaps.Client(key=googlemaps_key)

    # Used to "quickly" limit distance to a certain bounding box so we're not comparing distances for every city on Earth
    # Not particularly accurate, but good enough for the purpose
    def reverse_haversine(start_location, dist_km, direction="N"):
        dir_lookup = {
            "N": 0,
            "E": math.pi/2,
            "S": math.pi,
            "W": -math.pi/2,
        }
        result = np.radians(start_location)
        lat, long = result
        dist = dist_km / pgeocode.EARTH_RADIUS
        theta = dir_lookup[direction]  # Direction in radians

        lat2 = math.asin(
            (math.sin(lat) * math.cos(dist)) + (math.cos(lat) * math.sin(dist) * math.cos(theta))
        )
        long2 = (
            long + math.atan2((math.sin(theta) * math.sin(dist) * math.cos(lat)),
            (math.cos(dist) - (math.sin(lat) * math.sin(lat2))))
        )

        return math.degrees(lat2), math.degrees(long2)

    loc = (start_lat, start_lon)
    bounds = {
        "north": reverse_haversine(loc, flight_radius_km * 2, "N")[0],
        "east": reverse_haversine(loc, flight_radius_km * 2, "E")[1],
        "south": reverse_haversine(loc, flight_radius_km * 2, "S")[0],
        "west": reverse_haversine(loc, flight_radius_km * 2, "W")[1],
    }

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
        ]
    )

    proj_df = gpd.GeoDataFrame(
        city_df,
        geometry=gpd.points_from_xy(city_df.latitude, city_df.longitude),
        crs=pyproj.CRS(EPSG_STR),
    )

    # Quick and dirty filter to filter out most of the cities that are outside the bounds of the area to reduce computation
    proj_df = proj_df.query(
        f"not (latitude > {bounds['north']} or longitude > {bounds['east']} or latitude < {bounds['south']} or longitude < {bounds['west']})"
    )

    transformer = pyproj.Transformer.from_proj(EPSG_STR, AEQD_STR)

    proj_df["geometry"] = proj_df["geometry"].transform(lambda i: Point(transformer.transform(i.x, i.y)))
    proj_df["distance"] = proj_df["geometry"].distance(Point(0, 0))

    closest_cities = (
        proj_df.query(f"distance > {disaster_radius_km} and distance <= {flight_radius_km} and `feature code` != 'PPL'")
    )

    for extra_filter in extra_filters:
        closest_cities = closest_cities.query(extra_filter)

    closest_cities = closest_cities.sort_values("population", ascending=False).head(60)

    if not os.path.exists("output"):
        os.mkdir("output")

    closest_cities.to_csv("output/closest_cities.txt")

    import itertools
    destinations = []
    iterrows = closest_cities.iterrows()
    with open("output/distance_matrix.json", "w") as matrix_file:
        while True:
            rows = list(itertools.islice(iterrows, 20))
            if not rows:
                break

            destination_set = [
                {
                    "name": city_data.asciiname,
                    "location": (city_data.latitude, city_data.longitude),
                }
                for _, city_data in rows
            ]

            dest_locations = [destination["location"] for destination in destination_set]

            destinations.extend(destination_set)

            distances = gmaps.distance_matrix(
                origins=loc, destinations=dest_locations,
                mode="driving", language="en", units="metric",
            )

            json.dump(distances, matrix_file)
            matrix_file.write("\n")

            for destination, address, distance, in zip(
                    destination_set,
                    distances["destination_addresses"],
                    distances["rows"][0]["elements"]
            ):
                destination["address"] = address
                destination["distance"] = distance.get("distance", {"text": "NA", "value": 999999})
                destination["duration"] = distance.get("duration", {"text": "NA", "value": 999999})
    sorted_destinations = list(sorted(destinations, key=lambda obj: obj["duration"]["value"]))[0:20]

    for destination in sorted_destinations:
        directions_result = gmaps.directions(
            (start_lat, start_lon),
            destination["location"],
            mode="driving")
        destination["route"] = directions_result

    with open('output/routes.json', 'w') as f:
        f.write(json.dumps({
            destination['name']: destination['route']
            for destination in sorted_destinations
        }))

    # mapping and shape utils
    import folium
    from folium import plugins

    # Create Map
    map = folium.Map(location=[start_lat, start_lon], zoom_start=7)

    start_m = folium.Marker([start_lat, start_lon], popup=(start_lat, start_lon),
                            icon=folium.Icon(icon='glyphicon glyphicon-fire', color='darkred'))
    start_m.add_to(map)

    # Plot conflict starting points
    for destination in sorted_destinations:
        loc_m = folium.Marker(destination["location"], popup=destination["name"],
                              icon=folium.Icon(icon='glyphicon glyphicon-home', color='blue'))
        loc_m.add_to(map)

        route = destination["route"]
        distance = route[0]['legs'][0]['distance']['text']
        duration = route[0]['legs'][0]['duration']['text']
        tooltip = f"Travel between <b>{start_lat} {start_lon}</b> and <b>{destination.get('name', 'N/A')}" \
                  f"</b> by car is <b>{distance}</b> and takes <b>{duration}</b>."
        polyline_ = polyline.decode(route[0]['overview_polyline']['points'])
        polyline_m = folium.PolyLine(polyline_, color='blue', tooltip=tooltip, weight=5)
        polyline_m.add_to(map)

    # Add fullscreen button
    plugins.Fullscreen().add_to(map)
    map.save("output/routes.html")


if __name__ == "__main__":

    description = """
    Model that provides potential evacuation routes
    """

    arg_parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    arg_parser.add_argument(
        "start_lat",
        type=float,
        help="",
    )
    arg_parser.add_argument(
        "start_lon",
        type=float,
        help="",
    )
    arg_parser.add_argument(
        "disaster_radius_km",
        type=float,
        help="",
    )
    arg_parser.add_argument(
        "flight_radius_km",
        type=float,
        help="",
    )
    arg_parser.add_argument(
        "--travel-mode",
        type=str,
        help="Sets the name of the file that is output",
        choices=list(TravelModes.__members__.keys()),
        default='Driving',
    )
    arg_parser.add_argument(
        "--extra-filters",
        type=str,
        default="[]",
    )
    args = arg_parser.parse_args()

    find_cities(
        start_lat=args.start_lat,
        start_lon=args.start_lon,
        disaster_radius_km=args.disaster_radius_km,
        flight_radius_km=args.flight_radius_km,
        travel_mode=TravelModes[args.travel_mode],
        extra_filters=json.loads(args.extra_filters),
    )
