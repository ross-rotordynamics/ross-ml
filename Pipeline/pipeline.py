"""ROSS Machine Learning.

ROSS-ML is a module to create neural networks aimed at calculating rotodynamic
coefficients for bearings and seals.
"""
# fmt: off
import os
import webbrowser
from pickle import dump, load

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import (ks_2samp, ttest_ind)

from sklearn.metrics import (explained_variance_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from statsmodels.distributions.empirical_distribution import ECDF
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, load_model

from plotly import graph_objects as go
from plotly.subplots import make_subplots
from plotly.figure_factory import create_scatterplotmatrix

# fmt: on

__all__ = ["HTML_formater", "Pipeline", "Model", "PostProcessing"]


def HTML_formater(df, name):
    html_string = """
                    <html>
                    <head><title>HTML Pandas Dataframe with CSS</title></head>
                    <link rel="stylesheet" type="text/css" href="css/panda_style.css"/>
                   <body>
                       {table}
                   </body>
                    </html>.
               """
    with open("tables/{}.html".format(name), "w") as f:
        f.write(html_string.format(table=df.to_html(classes="mystyle")))


class Pipeline:
    """

    Parameters
    ----------
    df : pd.Dataframe

    Examples
    --------
    >>> import pandas as pd
    >>> import rossml as rsml

    Importing and collecting data
    >>> df = pd.read_csv('seal_fake.csv')
    >>> df_val= pd.read_csv('xllaby_data-componentes.csv')

    >>> df_val.fillna(df.mean)
    >>> D = Pipeline(df)
    >>> D.set_features(0, 20)
    >>> D.set_labels(20, len(D.df.columns))
    >>> D.feature_reduction(15)
    >>> D.data_scaling(0.1, scalers=[RobustScaler(), RobustScaler()], scaling=True)
    >>> D.build_Sequential_ANN(4, [50, 50, 50, 50])
    >>> model, predictions = D.model_run(batch_size=300, epochs=1000)

    Get the model configurations to change it afterwards
    >>> # model.get_config()
    >>> D.model_history()
    >>> D.metrics()

    Post-processing data
    >>> postproc = PostProcessing(D.train,D.test)
    >>> postproc.plot_overall_results()
    >>> postproc.plot_confidence_bounds(a = 0.01)
    >>> postproc.plot_standardized_error()
    >>> postproc.plot_qq()

    Displays the HTML report
    >>> # postproc.show()
    >>> url = 'results'
    >>> D.hypothesis_test()
    >>> D.save_model('teste')
    >>> model = Model('teste')
    >>> model.load_model()
    >>> X = Pipeline(df_val).set_features(0,20)
    >>> results = model.predict(X)
    """

    def __init__(self, df):
        path1 = "img"
        path2 = "tables"
        if not os.path.isdir(path1):
            os.makedirs(path1)
        if not os.path.isdir(path2):
            os.makedirs(path2)
        self.df = df
        self.df.dropna(inplace=True)

    def set_features(self, start, end):
        """

        Parameters
        ----------
        start : int
            Start column of dataframe features
        end : TYPE
            End column of dataframe features

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.X = self.df[self.df.columns[start:end]]
        return self.X

    def set_labels(self, start, end):
        """

        Parameters
        ----------
        start : TYPE
            Start column of dataframe labels
        end : TYPE
            End column of dataframe labels

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.y = self.df[self.df.columns[start:end]]
        return self.y

    def feature_reduction(self, n):
        """

        Parameters
        ----------
        n : int
            Number of relevant features

        Returns
        -------
        Minimum number of features that satisfies "n" for each label.

        """
        # define the model
        model = DecisionTreeRegressor()

        # fit the model
        model.fit(self.X, self.y)

        # get importance
        importance = model.feature_importances_

        # summarize feature importance
        featureScores = pd.concat(
            [pd.DataFrame(self.X.columns), pd.DataFrame(importance)], axis=1
        )
        featureScores.columns = ["Specs", "Score"]
        self.best = featureScores.nlargest(n, "Score")["Specs"].values
        self.X = self.X[self.best]

        return self.X

    def data_scaling(self, test_size, scaling=True, scalers=None):
        """

        Parameters
        ----------
        test_size : float
            Percentage of data destined for testing.
        scalers : scikit-learn object
            scikit-learn scalers
        scaling : boolean, optional
            Choose between scaling the data or not.
            The default is True.

        Returns
        -------
        x_train : array
            Features destined for training.
        x_test : array
            Features destined for test.
        y_train : array
            Labels destined for training.
        y_test : array
             Labels destined for test.

        """
        if scalers is None:
            scalers = []

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size
        )
        if scaling:
            if len(scalers) >= 1:
                self.scaler1 = scalers[0]
                self.x_train = self.scaler1.fit_transform(self.x_train)
                self.x_test = self.scaler1.transform(self.x_test)
                if len(scalers) == 2:
                    self.scaler2 = scalers[1]
                    self.y_train = self.scaler2.fit_transform(self.y_train)
                    self.y_test = self.scaler2.transform(self.y_test)
        else:
            self.scaler1 = None
            self.scaler2 = None

        return self.x_train, self.x_test, self.y_train, self.y_test

    def build_Sequential_ANN(self, hidden, neurons, dropout_layers=[], dropout=[]):
        """

        Parameters
        ----------
        hidden : int
            Number of hidden layers.
        neurons : list
            Number of neurons per layer.
        dropout_layers : list, optional
            Dropout layers position.
            The default is [].
        dropout : list, optional
            List with dropout values.
            The default is [].

        Returns
        -------
        model : keras neural network
        """
        if dropout_layers is None:
            dropout_layers = []
        if dropout is None:
            dropout = []

        self.model = Sequential()
        self.model.add(Dense(len(self.X.columns), activation="relu"))
        j = 0  # Dropout counter
        for i in range(hidden):
            if i in dropout_layers:
                self.model.add(Dropout(dropout[j]))
                j += 1
            self.model.add(Dense(neurons[i], activation="relu"))
        self.model.add(Dense(len(self.y.columns)))
        self.config = self.model.get_config()

        return self.model

    def model_run(self, optimizer="adam", loss="mse", batch_size=16, epochs=500):
        """

        Parameters
        ----------
        optimizer : string, optional
            Choose a optimizer. The default is 'adam'.
        loss : string, optional
            Choose a loss. The default is 'mse'.
        batch_size : int, optional
            batch_size . The default is 16.
        epochs : int, optional
            Choose number of epochs. The default is 500.

        Returns
        -------
        model : keras neural network
        predictions :
        """
        self.model.compile(optimizer=optimizer, loss=loss)
        self.history = self.model.fit(
            x=self.x_train,
            y=self.y_train,
            validation_data=(self.x_test, self.y_test),
            batch_size=batch_size,
            epochs=epochs,
        )
        self.predictions = self.model.predict(self.x_test)
        self.train = pd.DataFrame(
            self.scaler2.inverse_transform(self.predictions), columns=self.y.columns
        )
        self.test = pd.DataFrame(
            self.scaler2.inverse_transform(self.y_test), columns=self.y.columns
        )
        return self.model, self.predictions

    def model_history(self):
        """Plot model history.

        Examples
        --------
        """
        hist = pd.DataFrame(
            self.D.history.history
        )
        axes_default = dict(
            gridcolor="lightgray",
            showline=True,
            linewidth=1.5,
            linecolor="black",
            mirror=True,
            exponentformat="none",
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=list(range(hist["loss"].size)),
                y=np.log10(hist["loss"]),
                mode="lines",
                line=dict(color="blue"),
                name="Loss",
                legendgroup="Loss",
                hoverinfo="none",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(hist["val_loss"].size)),
                y=np.log10(hist["val_loss"]),
                mode="lines",
                line=dict(color="orange"),
                name="Val_loss",
                legendgroup="Val_loss",
                hoverinfo="none",
            )
        )

        fig.update_xaxes(
            title=dict(text="Epoch", font=dict(size=15)), **axes_default,
        )
        fig.update_yaxes(
            title=dict(text="Log<sub>10</sub>Loss", font=dict(size=15)), **axes_default,
        )
        fig.update_layout(
            plot_bgcolor="white",
            legend=dict(bgcolor="white", bordercolor="black", borderwidth=1,),
        )
        fig.write_html(r"img\history.html")

        return fig


    def metrics(self):
        """Print model metrics.

        This function displays the model metrics while the neural network is being
        built.

        Prints
        ------
        The mean absolute error (MAE).
        The mean squared error (MSE).
        The coefficient of determination (R-squared).
        The adjusted coefficient of determination (adjusted R-squared).
        The explained variance (discrepancy between a model and actual data).
        """
        R2_a = 1 - (
            (len(self.predictions) - 1)
            * (1 - r2_score(self.y_test, self.predictions))
            / (len(self.predictions) - (1 + len(self.X.columns)))
        )
        MAE = mean_absolute_error(self.y_test, self.predictions)
        MSE = mean_squared_error(self.y_test, self.predictions)
        R2 = r2_score(self.y_test, self.predictions)
        explained_variance = explained_variance_score(self.y_test, self.predictions)
        metrics = pd.DataFrame(
            np.round([MAE, MSE, R2, R2_a, explained_variance], 3),
            index=["MAE", "MSE", "R2", "R2_adj", "explained variance"],
            columns=["Metric"],
        )
        HTML_formater(metrics, "Metrics")
        print(
            "Scores:\nMAE: {}\nMSE: {}\nR^2:{}\nR^2 adjusted:{}\nExplained variance:{}".format(
                MAE, MSE, R2, R2_a, explained_variance
            )
        )

    def hypothesis_test(self, kind="ks", p_value=0.05):
        """Run a hypothesis test.

        Parameters
        ----------
        kind : string, optional
            Hypothesis test kind. Options are:
                "w": Welch test
                    Calculate the T-test for the means of two independent samples of
                    scores. This is a two-sided test for the null hypothesis that 2
                    independent samples have identical average (expected) values.
                    This test assumes that the populations have identical variances by
                    default.
                "ks": Komolgorov-Smirnov test
                    Compute the Kolmogorov-Smirnov statistic on 2 samples. This is a
                    two-sided test for the null hypothesis that 2 independent samples
                    are drawn from the same continuous distribution.
            The default is 'ks'.

            * See scipy.stats.ks_2samp and scipy.stats.ttest_ind documentation for more
            informations.

        p_value : float, optional
            Critical value. Must be within 0 and 1.
            The default is 0.05.

        Returns
        -------
        p_df : pd.DataFrame
            Hypothesis test results.

        Examples
        --------
        """
        if kind == "ks":
            p_values = np.round(
                [
                    ks_2samp(self.train[var], self.test[var]).pvalue
                    for var in self.train.columns
                ],
                3,
            )
            p_df = pd.DataFrame(
                p_values, index=self.test.columns, columns=["p-value: train"]
            )
            p_df["status"] = [
                "Not Reject H0" if p > p_value else "Reject H0"
                for p in p_df["p-value: train"]
            ]
            HTML_formater(p_df, "KS Test")
        elif kind == "w":
            p_values = np.round(
                [
                    ttest_ind(self.train[var], self.test[var], equal_var=False).pvalue
                    for var in self.train.columns
                ],
                3,
            )
            p_df = pd.DataFrame(
                p_values, index=self.train.columns, columns=["p-value: acc"]
            )
            p_df["status"] = [
                "Not Reject H0" if p > p_value else "Reject H0"
                for p in p_df["p-value: acc"]
            ]
            HTML_formater(p_df, "Welch Test")
            # p_df.to_html(r'tables\Welch test.html')
        return p_df

    def validation(self, x, y):
        """

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.

        Returns
        -------
        train : pd.DataFrame
            DESCRIPTION.
        test : pd.DataFrame
            DESCRIPTION.

        """
        self.test = y
        if self.scaler1 is not None:
            X_scaled = self.scaler1.transform(x.values.reshape(-1, len(x.columns)))
        if self.scaler2 is not None:
            self.train = pd.DataFrame(
                self.scaler2.inverse_transform(self.model.predict(X_scaled)),
                columns=y.columns,
            )
        else:
            self.train = pd.DataFrame(self.model.predict(X_scaled), columns=y.columns)
        return self.train, self.test

    def save_model(self, name):
        """

        Parameters
        ----------
        name : str
            DESCRIPTION.

        """
        if not os.path.isdir(name):
            os.makedirs(name)
        self.model.save(r"{}/{}.h5".format(name, name))
        dump(self.y.columns, open(r"{}/{}_columns.pkl".format(name, name), "wb"))
        dump(self.best, open(r"{}/{}_best_features.pkl".format(name, name), "wb"))
        if self.scaler1 != None:
            dump(self.scaler1, open(r"{}/{}_scaler1.pkl".format(name, name), "wb"))
        if self.scaler2 != None:
            dump(self.scaler2, open(r"{}/{}_scaler2.pkl".format(name, name), "wb"))


class Model:
    def __init__(self, name):
        self.name = name

    def load_model(self):
        self.model = load_model(r"{}/{}.h5".format(self.name, self.name))
        self.columns = load(
            open(r"{}/{}_columns.pkl".format(self.name, self.name), "rb")
        )
        # load best features
        try:
            self.best = load(
                open(r"{}/{}_best_features.pkl".format(self.name, self.name), "rb")
            )
        except:
            self.best = None
        # load the scaler
        try:
            self.scaler1 = load(
                open(r"{}/{}_scaler1.pkl".format(self.name, self.name), "rb")
            )
        except:
            self.scaler1 = None
        try:
            self.scaler2 = load(
                open(r"{}/{}_scaler2.pkl".format(self.name, self.name), "rb")
            )
        except:
            self.scaler2 = None
        # return self.model

    def predict(self, X):
        if self.best is None:
            self.X = X
        else:
            self.X = X[self.best]
        if self.scaler1 != None:
            self.X = self.scaler1.transform(self.X)
        else:
            self.X = X
        if self.scaler2 != None:
            predictions = self.model.predict(self.X)
            self.results = pd.DataFrame(
                self.scaler2.inverse_transform(predictions), columns=self.columns
            )
        else:
            self.results = pd.DataFrame(predictions, columns=self.columns)
        return self.results


class PostProcessing:
    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.test.index = self.train.index

    def plot_overall_results(self, save_fig=False):
        """

        Parameters
        ----------
        save_fig : bool, optional
            If save_fig is True, saves the result in img folder. If False, does not
            save. The default is True.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        df = pd.concat([self.train, self.test], axis=0)
        df["label"] = [
            "train" if x < len(self.train) else "test" for x in range(len(df))
        ]
        axes_default = dict(
            gridcolor="lightgray",
            showline=True,
            linewidth=1.5,
            linecolor="black",
            mirror=True,
            exponentformat="power",
            zerolinecolor="lightgray",
        )

        fig = create_scatterplotmatrix(
            df, diag="box", index="label", title="",
        )
        fig.update_layout(
            width=1500,
            height=1500,
            plot_bgcolor="white",
            hovermode=False,
        )
        fig.update_xaxes(**axes_default)
        fig.update_yaxes(**axes_default)

        if save_fig:
            fig.write_html(r"img\pairplot.html")

        return fig

    def plot_confidence_bounds(self, a=0.01, percentile=0.05, save_fig=False):
        """Plot a confidence interval based on DKW inequality.

        Parameters
        ----------
        a : float
            Significance level.A number between 0 and 1.
        percentile : float, optional
            Percentile to compute, which must be between 0 and 100 inclusive.
        save_fig : bool, optional
            If save_fig is True, saves the result in img folder. If False, does not
            save. The default is True.

        Returns
        -------
        figures : list
            List of figures plotting DKW inequality for each variable in dataframe.
        """
        P = np.arange(0, 100, percentile)
        en = np.sqrt((1 / (2 * len(P))) * np.log(2 / a))
        var = list(self.train.columns)

        axes_default = dict(
            gridcolor="lightgray",
            showline=True,
            linewidth=1.5,
            linecolor="black",
            mirror=True,
            exponentformat="power",
        )

        figures = []
        for v in var:
            F = ECDF(self.train[v])
            G = ECDF(self.test[v])
            Gu = [min(Gn + en, 1) for Gn in G.y]
            Gl = [max(Gn - en, 0) for Gn in G.y]

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=np.concatenate((G.x, G.x[::-1])),
                    y=np.concatenate((Gl, Gu[::-1])),
                    mode="lines",
                    line=dict(color="lightblue"),
                    fill="toself",
                    fillcolor="lightblue",
                    opacity=0.6,
                    name=f"{v} - {100 * (1 - percentile)}% Confidence Interval",
                    legendgroup=f"CI - {v}",
                    hoverinfo="none",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=G.x,
                    y=G.y,
                    mode="lines",
                    line=dict(color="darkblue", dash="dash"),
                    name=f"test {v}",
                    legendgroup=f"test {v}",
                    hoverinfo="none",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=F.x,
                    y=F.y,
                    mode="lines",
                    line=dict(color="magenta", width=2.0, dash="dot"),
                    name=f"test {v}",
                    legendgroup=f"train {v}",
                    hoverinfo="none",
                )
            )

            fig.update_xaxes(
                title=dict(text=f"{v}", font=dict(size=15)), **axes_default,
            )
            fig.update_yaxes(
                title=dict(text="Cumulative Distribution Function", font=dict(size=15)),
                **axes_default,
            )
            fig.update_layout(
                plot_bgcolor="white",
                legend=dict(bgcolor="white", bordercolor="black", borderwidth=1,),
            )
            figures.append(fig)

            if save_fig:
                fig.write_html(r"img\CI_{}.html".format(v))

        return figures

    def plot_qq(self, save_fig=False):
        """

        Parameters
        ----------
        save_fig : bool, optional
            If save_fig is True, saves the result in img folder. If False, does not
            save. The default is True.

        Returns
        -------
        figures : list
            List of figures plotting quantile-quantile for each variable in dataframe.
        """
        axes_default = dict(
            gridcolor="lightgray",
            showline=True,
            linewidth=1.5,
            linecolor="black",
            mirror=True,
            exponentformat="power",
        )

        figures = []
        var = list(self.train.columns)
        for v in var:
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=self.test[v],
                    y=self.train[v],
                    mode="markers",
                    marker=dict(size=8, color="orange"),
                    name=f"Test points - {v}",
                    legendgroup=f"Test points - {v}",
                    hovertemplate=(
                        f"{v}<sub>test</sub> : %{{x:.2f}}<br>{v}<sub>train</sub> : %{{y:.2f}}"
                    ),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=self.test[v],
                    y=self.test[v],
                    mode="lines",
                    line=dict(color="blue", dash="dash"),
                    name="Y<sub>test</sub> = Y<sub>train</sub>",
                    legendgroup="test_test",
                    hoverinfo="none",
                )
            )

            fig.update_xaxes(
                title=dict(text=f"{v} <sub>test</sub>", font=dict(size=15)),
                **axes_default,
            )
            fig.update_yaxes(
                title=dict(text=f"{v} <sub>train</sub>", font=dict(size=15)),
                **axes_default,
            )
            fig.update_layout(
                plot_bgcolor="white",
                legend=dict(bgcolor="white", bordercolor="black", borderwidth=1,),
            )
            figures.append(fig)

            if save_fig:
                fig.write_html(r"img\qq_plot_{}.html".format(v))

        return figures

    def plot_standardized_error(self, save_fig=False):
        """Plot and save the graphic for standardized error.

        Parameters
        ----------
        save_fig : bool, optional
            If save_fig is True, saves the result in img folder. If False, does not
            save. The default is False.

        Returns
        -------
        figures : list
            List of figures plotting standardized error for each variable in dataframe.
        """
        var = list(self.train.columns)
        error = self.test - self.train
        error = error / error.std()
        error.dropna(inplace=True)
        axes_default = dict(
            gridcolor="lightgray",
            showline=True,
            linewidth=1.5,
            linecolor="black",
            mirror=True,
            exponentformat="power",
        )

        figures = []
        for v in var:
            error = self.test[v] - self.train[v]
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=self.test[v],
                    y=error[v],
                    mode="markers",
                    marker=dict(size=8, color="orange"),
                    name=f"Test points - {v}",
                    legendgroup=f"Test points - {v}",
                    showlegend=True,
                    hovertemplate=(
                        f"{v}<sub>test</sub> : %{{x:.2f}}<br>error : %{{y:.2f}}"
                    ),
                )
            )

            fig.update_xaxes(
                title=dict(text=f"{v} <sub>test</sub>", font=dict(size=15)),
                **axes_default,
            )
            fig.update_yaxes(
                title=dict(text=f"Standardized Error", font=dict(size=15)),
                **axes_default,
            )
            fig.update_layout(
                plot_bgcolor="white",
                legend=dict(bgcolor="white", bordercolor="black", borderwidth=1,),
                shapes=[
                    dict(
                        type="line",
                        xref="paper",
                        x0=0,
                        x1=1,
                        yref="y",
                        y0=0,
                        y1=0,
                        line=dict(color="blue", dash="dash"),
                    )
                ],
            )
            figures.append(fig)

            if save_fig:
                fig.write_html(r"img\standardized_error_{}_plot.html".format(v))

        return figures

    def plot_residuals_resume(self, save_fig=False):
        """Plot and save the graphic for residuals distribution.

        Parameters
        ----------
        save_fig : bool, optional
            If save_fig is True, saves the result in img folder. If False, does not
            save. The default is False.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        N = self.train.shape[1]
        colors = [
            "hsl(" + str(h) + ",50%" + ",50%)" for h in np.linspace(0, 360, N + 1)
        ]
        error_std = (self.test - self.train) / (self.test - self.train).std()
        labels = list(self.train.columns)

        axes_default = dict(
            gridcolor="lightgray",
            showline=True,
            linewidth=1.5,
            linecolor="black",
            mirror=True,
            exponentformat="power",
        )

        fig = go.Figure(
            data=[
                go.Box(
                    x=error_std[labels[i]],
                    marker_color=colors[i],
                    jitter=0.2,
                    pointpos=0,
                    boxpoints="all",
                    name=labels[i],
                    hoverinfo="none",
                    orientation="h",
                )
                for i in range(N)
            ]
        )
        fig.update_xaxes(
            title=dict(text="Standard Deviation", font=dict(size=15)),
            range=[-10, 10],
            tickvals=[-10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10],
            tickmode="array",
            **axes_default,
        )
        fig.update_yaxes(**axes_default)

        n = 7
        shape_x = np.linspace(-3, 3, n)
        shape_color = ["green", "blue", "red", "black", "red", "blue", "green"]
        fig.update_layout(
            plot_bgcolor="white",
            legend=dict(bgcolor="white", bordercolor="black", borderwidth=1),
            shapes=[
                dict(
                    type="line",
                    xref="x",
                    x0=shape_x[i],
                    x1=shape_x[i],
                    yref="paper",
                    y0=0,
                    y1=1,
                    line=dict(color=shape_color[i], dash="dash"),
                    name=f"σ = {abs(shape_x[i])}",
                )
                for i in range(n)
            ],
        )

        if save_fig:
            fig.write_html(r"img\residuals_resume.html")

        return fig

    def show(self, name="results", **kwargs):
        """Display the HTML report on browser.

        The report contains a brief content about the neural network built.

        Parameters
        ----------
        name : srt, optional
            The report file name.
            The default is "results".
        kwargs : optional inputs to plot the confidence bounds.
            a : float
                Significance level.A number between 0 and 1.
            percentile : float, optional
                Percentile to compute, which must be between 0 and 100 inclusive.

        Returns
        -------
        HTML report
            A interactive HTML report.
        """
        _ = self.plot_overall_results(save_fig=True)
        _ = self.plot_confidence_bounds(a, percentilve, save_fig=True)
        _ = self.plot_qq(save_fig=True)
        _ = self.plot_standardized_error(save_fig=True)
        _ = self.plot_residuals_resume(save_fig=True)

        return webbrowser.open(f"{name}.html", new=2)
