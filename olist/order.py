import pandas as pd
import numpy as np
from olist.utils import haversine_distance
from olist.data import Olist


class Order:
    '''
    DataFrames containing all orders as index,
    and various properties of these orders as columns
    '''
    def __init__(self):
        self.data = Olist().get_data()

    def get_wait_time(self, is_delivered=True):
        orders = self.data["orders"].copy()

        # datetime dönüşümleri
        orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"], errors="coerce")
        orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"], errors="coerce")
        orders["order_estimated_delivery_date"] = pd.to_datetime(orders["order_estimated_delivery_date"], errors="coerce")

        # süreler (gün)
        wait_time = (orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]).dt.total_seconds() / 86400
        expected_wait_time = (orders["order_estimated_delivery_date"] - orders["order_purchase_timestamp"]).dt.total_seconds() / 86400

        out = pd.DataFrame({
            "order_id": orders["order_id"],
            "wait_time": wait_time,
            "expected_wait_time": expected_wait_time,
            "delay_vs_expected": wait_time - expected_wait_time,
            "order_status": orders["order_status"],
        })

        if is_delivered:
            out = out[(out["order_status"] == "delivered") & out["wait_time"].notna() & out["expected_wait_time"].notna()]

        return out

    def get_review_score(self):
        reviews = self.data["order_reviews"].copy()

        # review_score sayısal olsun
        reviews["review_score"] = pd.to_numeric(reviews["review_score"], errors="coerce")

        # bir order için birden fazla review olabilme ihtimaline karşı:
        # score = ortalama, dimler = herhangi bir 5 / herhangi bir 1 var mı?
        agg = reviews.groupby("order_id").agg(
            review_score=("review_score", "mean"),
            dim_is_five_star=("review_score", lambda s: int((s == 5).any())),
            dim_is_one_star=("review_score", lambda s: int((s == 1).any())),
        ).reset_index()

        return agg[["order_id", "dim_is_five_star", "dim_is_one_star", "review_score"]]

    def get_number_items(self):
        items = self.data["order_items"].copy()

        out = (items.groupby("order_id")
               .size()
               .reset_index(name="number_of_items"))

        return out

    def get_number_sellers(self):
        items = self.data["order_items"].copy()

        out = (items.groupby("order_id")["seller_id"]
               .nunique()
               .reset_index(name="number_of_sellers"))

        return out

    def get_price_and_freight(self):
        items = self.data["order_items"].copy()

        items["price"] = pd.to_numeric(items["price"], errors="coerce")
        items["freight_value"] = pd.to_numeric(items["freight_value"], errors="coerce")

        out = (items.groupby("order_id")[["price", "freight_value"]]
               .sum()
               .reset_index())

        return out

    # Optional
    def get_distance_seller_customer(self):
        orders = self.data["orders"][["order_id", "customer_id"]].copy()
        customers = self.data["customers"][["customer_id", "customer_zip_code_prefix"]].copy()
        sellers = self.data["sellers"][["seller_id", "seller_zip_code_prefix"]].copy()
        items = self.data["order_items"][["order_id", "seller_id"]].copy()
        geo = self.data["geolocation"].copy()

        # geolocation: zip_code_prefix bazında tek satıra indir
        geo_cols = ["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"]
        geo = geo[geo_cols].copy()
        geo = (geo.groupby("geolocation_zip_code_prefix", as_index=False)
                 .agg(geolocation_lat=("geolocation_lat", "mean"),
                      geolocation_lng=("geolocation_lng", "mean")))

        # müşteri koordinatları
        c = orders.merge(customers, on="customer_id", how="left")
        c = c.merge(
            geo.rename(columns={
                "geolocation_zip_code_prefix": "customer_zip_code_prefix",
                "geolocation_lat": "customer_lat",
                "geolocation_lng": "customer_lng",
            }),
            on="customer_zip_code_prefix",
            how="left"
        )

        # satıcı koordinatları (order_items üzerinden order-seller eşleşmesi)
        s = items.merge(sellers, on="seller_id", how="left")
        s = s.merge(
            geo.rename(columns={
                "geolocation_zip_code_prefix": "seller_zip_code_prefix",
                "geolocation_lat": "seller_lat",
                "geolocation_lng": "seller_lng",
            }),
            on="seller_zip_code_prefix",
            how="left"
        )

        # order_id üzerinden müşteri koordinatını seller tarafına ekle
        m = s.merge(c[["order_id", "customer_lat", "customer_lng"]], on="order_id", how="left")

        # mesafe (km) — haversine_distance fonksiyonunun imzasına göre argüman sırasını kullanıyoruz
        m["distance_seller_customer"] = m.apply(
            lambda r: haversine_distance(r["seller_lat"], r["seller_lng"], r["customer_lat"], r["customer_lng"])
            if pd.notna(r["seller_lat"]) and pd.notna(r["seller_lng"]) and pd.notna(r["customer_lat"]) and pd.notna(r["customer_lng"])
            else np.nan,
            axis=1
        )

        # bir order’da birden fazla seller olabileceği için ortalama al
        out = (m.groupby("order_id", as_index=False)["distance_seller_customer"]
                 .mean())

        return out

    def get_training_data(self, is_delivered=True, with_distance_seller_customer=False):
        df = self.get_wait_time(is_delivered=is_delivered)

        df = df.merge(self.get_review_score(), on="order_id", how="left")
        df = df.merge(self.get_number_items(), on="order_id", how="left")
        df = df.merge(self.get_number_sellers(), on="order_id", how="left")
        df = df.merge(self.get_price_and_freight(), on="order_id", how="left")

        if with_distance_seller_customer:
            df = df.merge(self.get_distance_seller_customer(), on="order_id", how="left")
        else:
            df["distance_seller_customer"] = np.nan

 #final
        cols = [
            "order_id", "wait_time", "expected_wait_time", "delay_vs_expected", "order_status",
            "dim_is_five_star", "dim_is_one_star", "review_score",
            "number_of_items", "number_of_sellers", "price", "freight_value",
            "distance_seller_customer"
        ]
        df = df[cols].dropna()

        return df
