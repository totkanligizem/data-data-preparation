from pathlib import Path
import pandas as pd


class Olist:
    def ping(self):
        return "pong"

    def get_data(self):
        csv_path = Path("~/.workintech/olist/data/csv").expanduser()

        file_paths = sorted(csv_path.glob("*.csv"))
        file_names = [p.name for p in file_paths]

        key_names = [
            name.replace("olist_", "").replace("_dataset.csv", "").replace(".csv", "")
            for name in file_names
        ]

        data = {key: pd.read_csv(path) for key, path in zip(key_names, file_paths)}
        return data
        