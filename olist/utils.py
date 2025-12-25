from math import radians, sin, cos, asin, sqrt


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Compute distance between two pairs of coordinates (lon1, lat1, lon2, lat2)
    See - (https://en.wikipedia.org/wiki/Haversine_formula)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))


def return_significative_coef(model):
    """
    Returns p_value and coef for significant variables (p_value < 0.05)
    from a statsmodels object.
    """
    p_values = model.pvalues.reset_index()
    p_values.columns = ["variable", "p_value"]

    coef = model.params.reset_index()
    coef.columns = ["variable", "coef"]

    return (
        p_values.merge(coef, on="variable")
        .query("p_value < 0.05")
        .sort_values(by="coef", ascending=False)
    )


def plot_kde_plot(df, variable, dimension):
    """
    Plot a side by side kdeplot for `variable`, split by `dimension`.
    """
    import seaborn as sns  # lazy import: seaborn yoksa sadece bu fonksiyon patlar

    g = sns.FacetGrid(df, hue=dimension, col=dimension)
    g.map(sns.kdeplot, variable)
    return g
    