import pandas as pd
import numpy as np
from olist.data import Olist
from olist.order import Order


class Seller:
    def __init__(self):
        # Import data only once
        olist = Olist()
        self.data = olist.get_data()
        self.order = Order()

    def get_seller_features(self):
        """
        Returns a DataFrame with:
        'seller_id', 'seller_city', 'seller_state'
        """
        sellers = self.data['sellers'].copy()
        sellers.drop('seller_zip_code_prefix', axis=1, inplace=True)
        sellers.drop_duplicates(inplace=True)  # There can be multiple rows per seller
        return sellers

    def get_seller_delay_wait_time(self):
        """
        Returns a DataFrame with:
        'seller_id', 'delay_to_carrier', 'wait_time'
        """
        # Get data
        order_items = self.data['order_items'].copy()
        orders = self.data['orders'].query("order_status=='delivered'").copy()

        ship = order_items.merge(orders, on='order_id')

        # Handle datetime
        ship.loc[:, 'shipping_limit_date'] = pd.to_datetime(ship['shipping_limit_date'])
        ship.loc[:, 'order_delivered_carrier_date'] = pd.to_datetime(ship['order_delivered_carrier_date'])
        ship.loc[:, 'order_delivered_customer_date'] = pd.to_datetime(ship['order_delivered_customer_date'])
        ship.loc[:, 'order_purchase_timestamp'] = pd.to_datetime(ship['order_purchase_timestamp'])

        # Compute delay and wait_time
        def delay_to_logistic_partner(d):
            days = np.mean(
                (d.order_delivered_carrier_date - d.shipping_limit_date) / np.timedelta64(24, 'h')
            )
            return days if days > 0 else 0

        def order_wait_time(d):
            days = np.mean(
                (d.order_delivered_customer_date - d.order_purchase_timestamp) / np.timedelta64(24, 'h')
            )
            return days

        delay = ship.groupby('seller_id').apply(delay_to_logistic_partner).reset_index()
        delay.columns = ['seller_id', 'delay_to_carrier']

        wait = ship.groupby('seller_id').apply(order_wait_time).reset_index()
        wait.columns = ['seller_id', 'wait_time']

        df = delay.merge(wait, on='seller_id')
        return df

    def get_active_dates(self):
        """
        Returns a DataFrame with:
        'seller_id', 'date_first_sale', 'date_last_sale', 'months_on_olist'
        """
        # First, get only orders that are approved
        orders_approved = self.data['orders'][['order_id', 'order_approved_at']].dropna()

        # Create a (orders <> sellers) join table
        orders_sellers = orders_approved.merge(self.data['order_items'], on='order_id')[
            ['order_id', 'seller_id', 'order_approved_at']
        ].drop_duplicates()

        orders_sellers["order_approved_at"] = pd.to_datetime(orders_sellers["order_approved_at"])

        df = orders_sellers.groupby('seller_id').agg({
            "order_approved_at": ["min", "max"]
        })
        df.columns = ["date_first_sale", "date_last_sale"]

        df['months_on_olist'] = round(
            (df['date_last_sale'] - df['date_first_sale']) / np.timedelta64(30, 'D')
        )

        df = df.reset_index()
        return df

    def get_quantity(self):
        """
        Returns a DataFrame with:
        'seller_id', 'n_orders', 'quantity', 'quantity_per_order'
        """
        order_items = self.data['order_items']

        n_orders = order_items.groupby('seller_id')['order_id'].nunique().reset_index()
        n_orders.columns = ['seller_id', 'n_orders']

        quantity = order_items.groupby('seller_id', as_index=False).agg({'order_id': 'count'})
        quantity.columns = ['seller_id', 'quantity']

        result = n_orders.merge(quantity, on='seller_id')
        result['quantity_per_order'] = result['quantity'] / result['n_orders']
        return result

    def get_sales(self):
        """
        Returns a DataFrame with:
        'seller_id', 'sales'
        """
        return (
            self.data['order_items'][['seller_id', 'price']]
            .groupby('seller_id')
            .sum()
            .rename(columns={'price': 'sales'})
            .reset_index()
        )

    def get_review_score(self):
        """
        Returns a DataFrame with:
        'seller_id', 'share_of_five_stars', 'share_of_one_stars', 'review_score'
        """
        reviews = self.data["order_reviews"][["order_id", "review_score"]].dropna().copy()
        seller_orders = self.data["order_items"][["order_id", "seller_id"]].drop_duplicates()

        df = seller_orders.merge(reviews, on="order_id", how="inner")

        df["dim_is_one_star"] = (df["review_score"] == 1).astype(int)
        df["dim_is_five_star"] = (df["review_score"] == 5).astype(int)

        out = df.groupby("seller_id", as_index=False).agg({
            "dim_is_one_star": "mean",
            "dim_is_five_star": "mean",
            "review_score": "mean"
        })

        out.columns = ["seller_id", "share_of_one_stars", "share_of_five_stars", "review_score"]
        return out

    def get_training_data(self):
        """
        Returns a DataFrame with:
        ['seller_id', 'seller_city', 'seller_state', 'delay_to_carrier',
        'wait_time', 'date_first_sale', 'date_last_sale', 'months_on_olist',
        'share_of_one_stars', 'share_of_five_stars', 'review_score',
        'n_orders', 'quantity', 'quantity_per_order', 'sales']
        """
        training_set = (
            self.get_seller_features()
            .merge(self.get_seller_delay_wait_time(), on='seller_id')
            .merge(self.get_active_dates(), on='seller_id')
            .merge(self.get_quantity(), on='seller_id')
            .merge(self.get_sales(), on='seller_id')
        )

        review_df = self.get_review_score()
        if review_df is not None:
            training_set = training_set.merge(review_df, on='seller_id', how='left')

        return training_set
        