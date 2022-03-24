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

class TravelModes(Enum):
    Driving = "DRIVING"
    Walking = "WALKING"

CITY_FILE = "cities1000.txt"


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

    closest_cities = closest_cities.sort_values("population", ascending=False).head(20)

    if not os.path.exists("output"):
        os.mkdir("output")

    closest_cities.to_csv("output/closest_cities.txt")




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
