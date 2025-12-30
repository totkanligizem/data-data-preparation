from pathlib import Path

import pandas as pd


class Olist:
    def ping(self):
        return "pong"

    def get_data(self):
        csv_path = Path("~/.workintech/olist/data/csv").expanduser()

        file_paths = sorted(csv_path.glob("*.csv"))

        def clean_key(p: Path) -> str:
            return (
                p.name.replace("olist_", "")
                .replace("_dataset.csv", "")
                .replace(".csv", "")
            )

        # 1) Oku
        data = {clean_key(p): pd.read_csv(p) for p in file_paths}

        # 2) Key sırasını testin beklediği sıraya sabitle
        expected_keys = [
            "customers",
            "geolocation",
            "order_items",
            "order_payments",
            "order_reviews",
            "orders",
            "product_category_name_translation",
            "products",
            "sellers",
        ]
        data = {k: data[k] for k in expected_keys}

        # 3) Sellers kolon sırasını testin beklediği sıraya sabitle
        data["sellers"] = data["sellers"].loc[
            :, ["seller_city", "seller_id", "seller_state", "seller_zip_code_prefix"]
        ]

        return data
        