import argparse
import json
import os

from simple_refugee_route_model.evacuation import find_routes

if __name__ == "__main__":
    description = """
    Model that provides potential evacuation routes
    """

    arg_parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    arg_parser.add_argument(
        "config_file",
        type=str,
        help="Path to json config file. (E.g.: config.json)",
    )
    args = arg_parser.parse_args()
    config_file = args.config_file
    config: dict = json.load(open(config_file))
    googlemaps_key = config.pop("GOOGLEMAPS_KEY")
    os.environ["GOOGLEMAPS_KEY"] = googlemaps_key

    find_routes(
        **{
            key: value for key, value in config.items()
            if value is not None
        }
    )
