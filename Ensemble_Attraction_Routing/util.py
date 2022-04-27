import json
import numpy as np
import folium
import math
from fuzzywuzzy import fuzz, process

# Helper Encoder for json
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


basemaps = {
    "Google Maps": folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Maps",
        overlay=True,
        control=True,
    ),
    "Google Satellite": folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Satellite",
        overlay=True,
        control=True,
    ),
    "Google Terrain": folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Terrain",
        overlay=True,
        control=True,
    ),
    "Google Satellite Hybrid": folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Satellite",
        overlay=True,
        control=True,
    ),
    "Esri Satellite": folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite",
        overlay=True,
        control=True,
    ),
}


def get_closest(loc_lat, loc_lon, targets, mode, attraction_weight, attractions, gmaps):
    chunk_size = 25
    list_targets = [
        targets[i : i + chunk_size] for i in range(0, targets.shape[0], chunk_size)
    ]
    output = None
    closest_seconds = 100000000000
    closest_loc = None
    for i in list_targets:
        print(f'running distance matrix {i}')
        results = gmaps.distance_matrix(
            origins=[(loc_lat, loc_lon)],
            destinations=list(tuple(zip(i.latitude, i.longitude))),
            mode=mode,
        )
        largest_duration=0
        for idx, val in enumerate(results["rows"][0]["elements"]):
            if val["duration"]['value'] > largest_duration:
                largest_duration=val["duration"]['value']

        for idx, val in enumerate(results["rows"][0]["elements"]):
            if val["status"] == "ZERO_RESULTS":
                continue

            # get best matching name of the country in case they are slightly different - if names don't line up we have an issue
            country, ratio, idx = process.extractOne(i.iloc[idx]["country"], attractions["country"])

            attraction = attractions[
                attractions["country"] == country
            ].predicted_shares.iloc[0]

            seconds = (val["duration"]['value']/largest_duration) * (1 - attraction_weight) + ( 1 / math.sqrt(attraction)) * attraction_weight
            if seconds <= closest_seconds:
                closest_seconds = seconds
                closest_loc = i.iloc[idx]
                output = val

    return closest_loc, output





colors_ = [
    "lightblue",
    "orange",
    "lightred",
    "darkpurple",
    "darkgreen",
    "darkblue",
    "lightgray",
    "black",
    "cadetblue",
    "pink",
    "beige",
    "darkred",
    "lightgreen",
    "green",
    "red",
    "white",
    "blue",
    "purple",
    "gray",
]


def add_legend(map):
    legend_html = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100;300;400&display=swap');
    </style>

     <div style="
     padding-left:5px; padding-top:5px;
     position: fixed; 
     bottom: 50px; left: 50px; width: 160px; height: 120px;   
     border:2px solid grey; z-index:9999; border-radius: 15px;

     background-color:white;
     opacity: .85;

     font-size:14px;
     font-weight: bold;
     font-family: 'Roboto', sans-serif;
     ">

     <div class="awesome-marker-icon-darkred awesome-marker" style="margin-top: 10px; margin-left:5px;">
         <i class="fa-rotate-0 glyphicon glyphicon-glyphicon glyphicon-fire icon-white"></i>
     </div>
     <div style="margin-left:40px; margin-top:20px">Conflict Area</div>

     <div class="awesome-marker-icon-gray awesome-marker" style="margin-top: 60px; margin-left:5px;">
         <i class="fa-rotate-0 glyphicon glyphicon-glyphicon glyphicon-road icon-white"></i>
     </div>
     <div style="margin-left:40px; margin-top:25px">Destination Crossing</div>     


      </div> """.format(title="Legend html")
    map.get_root().html.add_child(folium.Element(legend_html))
    return map


def get_exit_route(row, mode, conflict_exit_routes):
    lat=None
    lng=None
    dest=None
    try:
        dest = conflict_exit_routes[row['#name']]['crossing']['country']
        lat = conflict_exit_routes[row['#name']]['crossing']['latitude']
        lng = conflict_exit_routes[row['#name']]['crossing']['longitude']
    except Exception as e:
        print(e)
        dest = None
    row[f'{mode}_destination'] = dest
    row[f'latitude'] = lat
    row[f'longitude'] = lng
    return row