import pandas as pd
import numpy as np
from olist.data import Olist
from olist.order import Order


class Product:
    def __init__(self):
        # Import data only once
        olist = Olist()
        self.data = olist.get_data()
        self.order = Order()

    def get_product_features(self):
        """
        Returns a DataFrame with:
        'product_id', 'category', 'product_name_length',
        'product_description_length', 'product_photos_qty', 'product_weight_g',
        'product_length_cm', 'product_height_cm', 'product_width_cm'
        """
        products = self.data['products']

        # (optional) convert category name to English
        en_category = self.data['product_category_name_translation']
        df = products.merge(en_category, on='product_category_name', how='left')

        # rename category col
        df = df.drop(columns=['product_category_name'])
        df = df.rename(columns={'product_category_name_english': 'category'})

        # Some Olist datasets have typos: "lenght" instead of "length"
        rename_map = {}
        if 'product_name_lenght' in df.columns:
            rename_map['product_name_lenght'] = 'product_name_length'
        if 'product_description_lenght' in df.columns:
            rename_map['product_description_lenght'] = 'product_description_length'
        df = df.rename(columns=rename_map)

        return df

    def get_price(self):
        """
        Return a DataFrame with:
        'product_id', 'price'
        """
        order_items = self.data['order_items']
        return (
            order_items[['product_id', 'price']]
            .groupby('product_id', as_index=True)
            .mean()
        )

    def get_wait_time(self):
        """
        Returns a DataFrame with:
        'product_id', 'wait_time'
        """
        orders_wait_time = self.order.get_wait_time()
        orders_products = self.data['order_items'][['order_id', 'product_id']].drop_duplicates()
        orders_products_with_time = orders_products.merge(orders_wait_time, on='order_id', how='inner')

        return (
            orders_products_with_time
            .groupby('product_id', as_index=False)
            .agg({'wait_time': 'mean'})
        )

    def get_review_score(self):
        """
        Returns a DataFrame with:
        'product_id', 'share_of_five_stars', 'share_of_one_stars',
        'review_score'
        """
        orders_reviews = self.order.get_review_score()
        orders_products = self.data['order_items'][['order_id', 'product_id']].drop_duplicates()
        df = orders_products.merge(orders_reviews, on='order_id', how='inner')

        result = df.groupby('product_id', as_index=False).agg({
            'dim_is_one_star': 'mean',
            'dim_is_five_star': 'mean',
            'review_score': 'mean',
        })
        result.columns = [
            'product_id', 'share_of_one_stars', 'share_of_five_stars', 'review_score'
        ]
        return result

    def get_quantity(self):
        """
        Returns a DataFrame with:
        'product_id', 'n_orders', 'quantity'
        """
        order_items = self.data['order_items']

        n_orders = (
            order_items.groupby('product_id')['order_id']
            .nunique()
            .reset_index()
        )
        n_orders.columns = ['product_id', 'n_orders']

        quantity = (
            order_items.groupby('product_id', as_index=False)
            .agg({'order_id': 'count'})
        )
        quantity.columns = ['product_id', 'quantity']

        return n_orders.merge(quantity, on='product_id')

    def get_sales(self):
        """
        Returns a DataFrame with:
        'product_id', 'sales'
        """
        return (
            self.data['order_items'][['product_id', 'price']]
            .groupby('product_id', as_index=True)
            .sum()
            .rename(columns={'price': 'sales'})
        )

    def get_training_data(self):
        """
        Returns a DataFrame with:
        ['product_id', 'product_name_length', 'product_description_length',
         'product_photos_qty', 'product_weight_g', 'product_length_cm',
         'product_height_cm', 'product_width_cm', 'category', 'wait_time',
         'price', 'share_of_one_stars', 'share_of_five_stars', 'review_score',
         'n_orders', 'quantity', 'sales']
        """
        training_set = (
            self.get_product_features()
            .merge(self.get_wait_time(), on='product_id', how='left')
            .merge(self.get_price(), on='product_id', how='left')
            .merge(self.get_review_score(), on='product_id', how='left')
            .merge(self.get_quantity(), on='product_id', how='left')
            .merge(self.get_sales(), on='product_id', how='left')
        )

        return training_set

    def get_product_cat(self, agg="mean"):
        """
        Returns a DataFrame with `category` as index, and aggregates numeric columns.
        - quantity: SUM (her zaman toplam satış adedi mantıklı)
        - diğer sayısal kolonlar: agg (mean/median vs.)
        """
        products = self.get_training_data()

        numeric_cols = list(products.select_dtypes(include=[np.number]).columns)

        # default: tüm sayısallar agg ile
        agg_params = {col: agg for col in numeric_cols}

        # ama quantity için toplam daha doğru
        if 'quantity' in agg_params:
            agg_params['quantity'] = 'sum'

        # product_id gibi anlamsız numeric varsa dışarı al (genelde object olur ama garanti)
        agg_params.pop('product_id', None)

        product_cat = products.groupby('category').agg(agg_params)

        return product_cat
        