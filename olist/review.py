import pandas as pd
import numpy as np
import math
from olist.data import Olist
from olist.order import Order


class Review:
    def __init__(self):
        # Import data only once
        olist = Olist()
        self.data = olist.get_data()
        self.order = Order()

    def get_review_length(self):
        """
        Returns a DataFrame with:
        'review_id', 'length_review', 'review_score'
        """
        reviews = self.data["order_reviews"].copy()

        # yorum metni bazen NaN olur -> boş string yap
        reviews["review_comment_message"] = reviews["review_comment_message"].fillna("")

        # uzunluk: karakter sayısı (string)
        reviews["length_review"] = reviews["review_comment_message"].astype(str).str.len()

        return reviews[["review_id", "length_review", "review_score"]]

    def get_main_product_category(self):
        """
        Returns a DataFrame with:
        'review_id', 'order_id', 'product_category_name'
        """
        reviews = self.data["order_reviews"][["review_id", "order_id"]].drop_duplicates()

        order_items = self.data["order_items"][["order_id", "product_id"]]
        products = self.data["products"][["product_id", "product_category_name"]]

        # order_id -> (ürün kategorileri)
        df = (
            reviews.merge(order_items, on="order_id", how="left")
                   .merge(products, on="product_id", how="left")
        )

        # Her review_id için en sık geçen kategori (mode). Eşitlik varsa alfabetik en küçüğünü seç.
        def _mode_category(s: pd.Series):
            s = s.dropna()
            if s.empty:
                return np.nan
            vc = s.value_counts()
            top = vc[vc == vc.max()].index
            return sorted(top)[0]

        main_cat = (
            df.groupby(["review_id", "order_id"], as_index=False)["product_category_name"]
              .agg(_mode_category)
        )

        return main_cat

    def get_training_data(self):
        """
        Returns a clean DataFrame (dropna) with at least:
        'review_id', 'order_id', 'review_score', 'length_review', 'product_category_name'
        """
        df_len = self.get_review_length()
        df_cat = self.get_main_product_category()

        df = df_cat.merge(df_len, on="review_id", how="left")

        # Eğitim datası: NaN’leri temizle (kategori veya skor yoksa uçur)
        df = df.dropna(subset=["review_score", "product_category_name"])

        # tip güvenliği
        df["review_score"] = df["review_score"].astype(int, errors="ignore")
        df["length_review"] = df["length_review"].fillna(0).astype(int, errors="ignore")

        return df
        