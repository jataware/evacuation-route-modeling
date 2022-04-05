import argparse
import csv
import datetime
from enum import Enum
import itertools
import json
import os

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import googlemaps
from haversine import inverse_haversine, Direction
import pyproj
import polyline


class TravelModes(Enum):
    Driving = "driving"
    Walking = "walking"
    Transit = "transit"

    @classmethod
    def travel_mode_text(cls, travel_mode):
        lookup = {
            cls.Driving.value: "by car",
            cls.Walking.value: "by foot",
            cls.Transit.value: "by transit",
        }
        return lookup.get(travel_mode, "")


CITY_FILE = "cities5000.txt"


def read_geonames_file(file_path):
    city_df = pd.read_csv(
        file_path,
        sep="\t",
        header=0,
        names=[
            "geonameid",
            "name",
            "name_ascii",
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
    return city_df


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


def find_routes(
        start_lat,
        start_lon,
        disaster_radius_km,
        flight_radius_km,
        travel_mode=TravelModes.Driving.value,
        extra_filters=[],
        destination_file=None,
        location_id_col="location_id",
        latitude_col="latitude",
        longitude_col="longitude",
):

    if destination_file is None:
        destination_file = CITY_FILE

    AEQD_STR = pyproj.Proj(f"+proj=aeqd +units=km +lat_0={start_lat} +lon_0={start_lon}")
    EPSG_STR = "EPSG:4326"

    googlemaps_key = os.environ.get("GOOGLEMAPS_KEY")
    gmaps = googlemaps.Client(key=googlemaps_key)

    start_loc = (start_lat, start_lon)

    bounds = {
        "north": inverse_haversine(start_loc, flight_radius_km * 2, Direction.NORTH)[0],
        "east": inverse_haversine(start_loc, flight_radius_km * 2, Direction.EAST)[1],
        "south": inverse_haversine(start_loc, flight_radius_km * 2, Direction.SOUTH)[0],
        "west": inverse_haversine(start_loc, flight_radius_km * 2, Direction.WEST)[1],
    }

    if destination_file.startswith("cities") and destination_file.endswith(".txt"):
        destination_df = read_geonames_file(CITY_FILE)
        location_id_col = "name_ascii"
    else:
        destination_df = pd.read_csv(destination_file)

    required_column_set = {location_id_col, latitude_col, longitude_col}

    if set(destination_df.columns).intersection(required_column_set) != required_column_set:
        raise ValueError(
            f"Datafile {destination_file} does not include the all required columns: {' '.join(required_column_set)}"
        )

    proj_df = gpd.GeoDataFrame(
        destination_df,
        geometry=gpd.points_from_xy(destination_df[latitude_col], destination_df[longitude_col]),
        crs=pyproj.CRS(EPSG_STR),
    )

    # Quick and dirty filter to filter out most of the cities that are outside the bounds of the area to reduce computation
    proj_df = proj_df.query(
        f"not ("
            f"{latitude_col} > {bounds['north']} or {longitude_col} > {bounds['east']} "
            f"or {latitude_col} < {bounds['south']} or {longitude_col} < {bounds['west']}"
        f")"
    )

    transformer = pyproj.Transformer.from_proj(EPSG_STR, AEQD_STR)

    proj_df["geometry"] = proj_df["geometry"].transform(lambda i: Point(transformer.transform(i.x, i.y)))
    proj_df["distance"] = proj_df["geometry"].distance(Point(0, 0))

    closest_cities = (
        proj_df.query(f"distance > {disaster_radius_km} and distance <= {flight_radius_km}")
    )

    for extra_filter in extra_filters:
        closest_cities = closest_cities.query(extra_filter)

    closest_cities = closest_cities.sort_values("distance", ascending=True).head(60)

    if not os.path.exists("output"):
        os.mkdir("output")
    if not os.path.exists("media"):
        os.mkdir("media")

    closest_cities.to_csv("output/closest_cities.txt")

    today = datetime.date.today().isoformat()
    destinations = []
    iterrows = closest_cities.iterrows()
    with open("output/distance_matrix.json", "w") as matrix_file:
        while True:
            rows = list(itertools.islice(iterrows, 20))
            if not rows:
                break

            destination_set = [
                {
                    "name": city_data[location_id_col],
                    "location": (city_data[latitude_col], city_data[longitude_col]),
                }
                for _, city_data in rows
            ]

            dest_locations = [destination["location"] for destination in destination_set]

            destinations.extend(destination_set)

            distances = gmaps.distance_matrix(
                origins=start_loc, destinations=dest_locations,
                mode=travel_mode, language="en", units="metric",
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
            mode=travel_mode)
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

    # Add evacuation area
    evacuation_area = folium.vector_layers.Circle(
        location=(start_lat, start_lon),
        radius=disaster_radius_km * 1000,
        color="#ff8888",
        fill=True,
        fill_opacity=0.3,
        popup=f"Evacuation distance: {disaster_radius_km} km"
    )
    evacuation_area.add_to(map)

    start_m = folium.Marker([start_lat, start_lon], popup=(start_lat, start_lon),
                            icon=folium.Icon(icon='glyphicon glyphicon-fire', color='darkred'))
    start_m.add_to(map)

    output_dataset = []
    travel_mode_desc = TravelModes.travel_mode_text(travel_mode)

    # Plot conflict starting points
    for destination in sorted_destinations:
        if not destination["route"]:
            continue
        route = destination["route"][0]
        leg = route['legs'][0]
        distance = leg['distance']['text']
        duration = leg['duration']['text']
        tooltip = f"Travel between <b>{start_lat} {start_lon}</b> and <b>{destination.get('name', 'N/A')}" \
                  f"</b> {travel_mode_desc} is <b>{distance}</b> and takes <b>{duration}</b>."

        popup_html = (
            f'''
            <div style="min-width: 400px">
                <h3>Travel to {destination["name"]} {travel_mode_desc}</h3>
                Total travel time: <b>{duration}</b><br/>
                Total travel distance: <b>{distance}</b>
            </div>
            '''
        )

        loc_m = folium.Marker(destination["location"], popup=folium.Popup(popup_html),
                              icon=folium.Icon(icon='glyphicon glyphicon-home', color='blue'))
        loc_m.add_to(map)

        polyline_ = polyline.decode(route['overview_polyline']['points'])
        polyline_m = folium.PolyLine(polyline_, color='blue', tooltip=tooltip, weight=5,
                                     popup=folium.Popup(popup_html))
        polyline_m.add_to(map)

        output_dataset.append([
            today,
            destination["name"],
            destination["location"][0],
            destination["location"][1],
            round(destination["duration"]["value"] / 3600, 3),
            round(destination["distance"]["value"] / 1000, 3),
            travel_mode,
        ])

    # Add fullscreen button
    plugins.Fullscreen().add_to(map)
    map.save("media/routes.html")

    with open("output/route_data.csv", "w") as output_datafile:
        output_csv = csv.writer(output_datafile, dialect="unix")
        output_csv.writerow([
            "date",
            "destination",
            "destination_latitude",
            "destination_longitude",
            "duration_hrs",
            "distance_km",
            "travel_mode",
        ])
        output_csv.writerows(output_dataset)


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
        "destination_file",
        nargs="?",
        type=str,
        default=None,
    )
    arg_parser.add_argument(
        "--travel-mode",
        type=str,
        help="Sets the name of the file that is output",
        choices=[travel_mode.value for travel_mode in TravelModes.__members__.values()],
        default=TravelModes.Driving.value,
    )
    arg_parser.add_argument(
        "--extra-filters",
        type=str,
        default="[]",
    )
    arg_parser.add_argument(
        "--location-id-col",
        type=str,
        help="Name of column in dataset that identifies the destination location identifier",
        default="location_id",
    )
    arg_parser.add_argument(
        "--latitude-col",
        help="Name of column in dataset that identifies the destination latitude",
        type=str,
        default="latitude",
    )
    arg_parser.add_argument(
        "--longitude-col",
        help="Name of column in dataset that identifies the destination longitude",
        type=str,
        default="longitude",
    )
    args = arg_parser.parse_args()

    find_routes(
        start_lat=args.start_lat,
        start_lon=args.start_lon,
        disaster_radius_km=args.disaster_radius_km,
        flight_radius_km=args.flight_radius_km,
        travel_mode=args.travel_mode,
        extra_filters=json.loads(args.extra_filters),
        destination_file=args.destination_file,
        location_id_col=args.location_id_col,
        latitude_col=args.latitude_col,
        longitude_col=args.longitude_col,
    )
