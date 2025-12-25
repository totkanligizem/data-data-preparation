## Olist Classes

This folder contains Olist Classes that handle the logic of data cleaning for our project.

For example, the below returns data as a Python dictionary using the `get_data` method from the `Olist` class:

```python
from olist.data import Olist
olist = Olist()
data = olist.get_data()
```

### Data

```python
from olist.data import Olist
```

Main methods:

- `get_data`: returns all Olist datasets as DataFrames within a Python dict.

### Order

```python
from olist.order import Order
```

Main method: 
- `get_training_data`: returns a DataFrame with: 
   - `order_id` (unique)
   - `wait_time`
   - `expected_wait_time`
   - `delay_vs_expected`
   -  `order_status`
   - `dim_is_five_star`
   - `dim_is_one_star`
   - `review_score`
   -  `number_of_products`
   - `number_of_sellers`
   - `price`
   - `freight_value`
   - `distance_seller_customer`

### Seller

```python
from olist.seller import Seller
```

Main method:
- `get_training_data`: returns a DataFrame with:
   - `seller_id` (unique)
   - `seller_city`
   - `seller_state`
   - `delay_to_carrier`
   - `wait_time`
   - `date_first_sale`
   - `date_last_sale`
   - `months_on_olist`
   - `share_of_one_stars`
   - `share_of_five_stars`
   - `review_score`
   - `n_orders`
   - `quantity`
   - `quantity_per_order`
   - `sales`

### Product

```python
from olist.product import Product
```

Main method:
- `get_training_data`: returns a DataFrame with 
   - `product_id` (unique)
   - `product_name_length`
   - `product_description_length`
   - `product_photos_qty`
   - `product_weight_g`
   - `product_length_cm`
   - `product_height_cm`
   - `product_width_cm`
   - `category`
   - `wait_time`
   - `price`
   - `share_of_one_stars`
   - `share_of_five_stars`
   - `review_score`
   - `n_orders`
   - `quantity`
   - `sales`

### Utils

Utility functions to help during the project.

```python
from olist.utils import *
```

- `haversine_distance(lat1, lng1, lat2, lng2)`: computes distance (in km) between two pairs of (lat, lng) [See Formula](https://en.wikipedia.org/wiki/Haversine_formula)
- `text_scatterplot(df, x, y)`: for a Dataframe `df`, creates a scatterplot with `x` and `y`. The index of `df` is the text label.
- `return_significative_coef(model)`: from a `model` as a statsmodels object, returns significant coefficients.
- `plot_kde_plot(df, variable, dimension)`: plots a side by side kdeplot from DataFrame `df` for `variable`, split by `dimension`.
