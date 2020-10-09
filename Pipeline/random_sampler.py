import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import linprog
from statsmodels.distributions.empirical_distribution import ECDF

__all__ = ["in_hull", "transforms", "mapping", "sampler"]


def in_hull(points, x):
    """


    Parameters
    ----------
    points : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T, np.ones((1, n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b, method="interior-point")

    return lp.success


def transforms(df, cols):
    """


    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    cols : TYPE
        DESCRIPTION.

    Returns
    -------
    CDF : TYPE
        DESCRIPTION.
    ICDF : TYPE
        DESCRIPTION.

    Examples
    --------
    >>> import pandas as pd
    >>> import rossml as rsml
    >>> df = pd.read_csv('xllaby_data-componentes.csv')
    >>> df.fillna(value=df.describe().loc["mean"], inplace=True)
    >>> CDF, ICDF = rsml.transforms(df, df.columns)
    """
    CDF = [ECDF(df[col]) for col in cols]
    ICDF = [
        interp1d(
            ECDF(df[col]).y,
            ECDF(df[col]).x,
            kind="zero",
            bounds_error=False,
            fill_value=(ECDF(df[col]).y[1], ECDF(df[col]).y[-2]),
        )
        for col in cols
    ]

    return CDF, ICDF


def mapping(df, transform):
    """


    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    transform : TYPE
        DESCRIPTION.

    Returns
    -------
    df_transf : TYPE
        DESCRIPTION.

    Examples
    --------
    >>> import pandas as pd
    >>> import rossml as rsml
    >>> df = pd.read_csv('xllaby_data-componentes.csv')
    >>> df.fillna(value=df.describe().loc["mean"], inplace=True)
    >>> df_U = rsml.mapping(df, CDF)
    >>> df_O = rsml.mapping(rsml.sampler(df_U, 500, 0.01), ICDF)

    """
    N = len(df)
    df_transf = pd.DataFrame(columns=df.columns)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if df.isnull().values.any():
        df.dropna(inplace=True)
        df = pd.concat([df, df.sample(n=N - len(df), replace=True)])

    for transf, var in zip(transform, df.columns):
        df_transf[var] = transf(df[var])

    df_transf.replace([np.inf, -np.inf], np.nan, inplace=True)
    if df_transf.isnull().values.any():
        df_transf.dropna(inplace=True)
        df_transf = pd.concat(
            [df_transf, df_transf.sample(n=N - len(df_transf), replace=True)]
        )

    return df_transf


def sampler(df, samples, frac):
    """


    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    samples : TYPE
        DESCRIPTION.
    frac : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    Examples
    --------
    >>> import pandas as pd
    >>> import rossml as rsml
    >>> df = pd.read_csv('xllaby_data-componentes.csv')
    >>> df_new = rsml.sampler(df, 500, 0.01)
    """
    samples = []
    for i in range(samples):
        weights = np.random.dirichlet(np.ones(int(frac * len(df))) / len(df))
        samples.append(
            df.sample(frac=int(frac * (len(df))) / len(df), replace=False).values.T.dot(
                weights
            )
        )

    return pd.DataFrame(np.array(samples), columns=df.columns)
