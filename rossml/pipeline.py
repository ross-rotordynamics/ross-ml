"""ROSS Machine Learning.

ROSS-ML is a module to create neural networks aimed at calculating rotodynamic
coefficients for bearings and seals.
"""
# fmt: off
import shutil
import webbrowser
from pathlib import Path
from pickle import dump, load

import numpy as np
import pandas as pd
from plotly import graph_objects as go
from plotly.figure_factory import create_scatterplotmatrix
from plotly.subplots import make_subplots
from scipy.stats import (chisquare, entropy, ks_2samp, normaltest, skew,
                         ttest_1samp, ttest_ind)
from sklearn.decomposition import PCA
from sklearn.feature_selection import (SelectKBest, f_regression,
                                       mutual_info_regression)
from sklearn.metrics import (explained_variance_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer,
                                   PowerTransformer, QuantileTransformer,
                                   RobustScaler, StandardScaler)
from sklearn.tree import DecisionTreeRegressor
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.stats.diagnostic import het_breuschpagan
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

# fmt: on

__all__ = [
    "HTML_formater",
    "available_models",
    "remove_model",
    "Pipeline",
    "Model",
    "PostProcessing",
]


def HTML_formater(df, name, file):
    """Table to HTML formater.

    This function takes a pandas DataFrame and writes it to HTML format. It's an
    auxiliary function to build the HTML Report.

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.
    name : str
        Neural network tag, indicating from which model it refers to.
    file : str
        The file name to save the DataFrame.
    """
    path = Path(__file__).parent
    html_string = """
                    <html>
                    <head><title>HTML Pandas Dataframe with CSS</title></head>
                    <link rel="stylesheet" type="text/css" href="css/panda_style.css"/>
                   <body>
                       {table}
                   </body>
                    </html>.
               """
    with open(path / f"models/{name}/tables/{file}.html", "w") as f:
        f.write(html_string.format(table=df.to_html(classes="mystyle")))


def available_models():
    """Check for available neural network models.

    This function returns a list of all neural network models saved.
    If None is available, it returns a message informing there's no models previously
    saved.

    Returns
    -------
    dirs : list
        List of all neural network models saved.

    Examples
    --------
    >>> import rossml as rsml
    >>> rsml.available_models()
    ['test_model']
    """
    try:
        path = Path(__file__).parent / "models"
        dirs = [folder.name for folder in path.iterdir() if folder.is_dir()]
        if len(dirs) == 0:
            dirs = "No neural network models available."
    except FileNotFoundError:
        dirs = "No neural network models available."

    return dirs


def remove_model(name):
    """Remove a previously saved network model.

    This function removes a neural network from "models" folder.

    Parameters
    ----------
    name : str
        The neural network folder's name to be deleted.

    Examples
    --------
    >>> import pandas as pd
    >>> import rossml as rsml
    >>> file = Path(__file__).parent / "tests/data/seal_data.csv"
    >>> df = pd.read_csv(file)

    Building the neural network model
    >>> name = "Model"
    >>> D = rsml.Pipeline(df, name)
    >>> sorted(rsml.available_models())
    ['Model', 'test_model']

    Removing a neural network
    >>> rsml.remove_model('Model')
    >>> rsml.available_models()
    ['test_model']
    """
    if isinstance(name, str):
        path = Path(__file__).parent / f"models/{name}"
    elif isinstance(name, Path):
        path = name
    for child in path.glob("*"):
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            remove_model(child)
    path.rmdir()


class Pipeline:
    r"""Generate an artificial neural netowrk.

    This class is a pipeline for building neural network models. From the data
    spreadsheet to the model it self, each function for this class has to be called in
    an exact order to guarantee its the correct functioning.

    The basic order to follow:
        - Pipeline
        - set_features()
        - set_labels()
        - feature_reduction()
        - data_scaling()
        - build_Sequential_ANN()
        - model_run()

    Parameters
    ----------
    df : pd.Dataframe

    name : str
        A tag for the neural network. This name is used to save the model, figures and
        tables within a folder named after this string.

    Examples
    --------
    >>> import pandas as pd
    >>> import rossml as rsml
    >>> from pathlib import Path
    >>> from sklearn.preprocessing import RobustScaler

    Importing and collecting data
    >>> file = Path(__file__).parent / "tests/data/seal_data.csv"
    >>> df = pd.read_csv(file)

    Building the neural network model
    >>> name = "Model"  # this name will be used to save your work
    >>> D = rsml.Pipeline(df, name)

    Selecting features and labels
    >>> features = D.set_features(1, 21)
    >>> labels = D.set_labels(21, len(D.df.columns))
    >>> new_features = D.feature_reduction(15)

    Data scaling and running the model
    >>> x_train, x_test, y_train, y_test = D.data_scaling(
    ...     0.1, scalers=[RobustScaler(), RobustScaler()], scaling=True
    ... )
    >>> model = D.build_Sequential_ANN(4, [50, 50, 50, 50])
    >>> model, predictions = D.model_run(batch_size=300, epochs=200)  # doctest: +ELLIPSIS
    Epoch 1/200...

    Get the model configurations to change it afterwards. These evaluations are
    important to decide whether the neural network meets or not the user requirements.
    >>> # model.get_config()
    >>> # fig = D.model_history()
    >>> # D.metrics()
    >>> df_test = D.hypothesis_test()

    Post-processing Data
    >>> results = PostProcessing(D.train, D.test, name)
    >>> fig = results.plot_overall_results()
    >>> fig = results.plot_confidence_bounds(a = 0.01)
    >>> fig = results.plot_standardized_error()
    >>> fig = results.plot_qq()

    Saving a model
    >>> # D.save()

    Displays the HTML report
    >>> # url = 'results'
    >>> # results.report(url)

    Loading a neural network model
    >>> # model = rsml.Model("Model")
    >>> # X = Pipeline(df).set_features(1, 21)
    >>> # results = model.predict(X)

    Removing models
    >>> rsml.remove_model('Model')
    """

    def __init__(self, df, name="Model"):
        path_model = Path(__file__).parent / f"models/{name}"
        path_img = Path(__file__).parent / f"models/{name}/img"
        path_table = Path(__file__).parent / f"models/{name}/tables"
        if not path_model.exists():
            path_model.mkdir()
            path_img.mkdir()
            path_table.mkdir()

        self.df = df
        self.df.dropna(inplace=True)
        self.name = name
        self.best = None

    def set_features(self, start, end):
        """Select the features from the input DataFrame.

        This methods takes the DataFrame and selects all the columns from "start"
        to "end" values to indicate which columns should be treated as features.

        Parameters
        ----------
        start : int
            Start column of dataframe features
        end : int
            End column of dataframe features

        Returns
        -------
        x : pd.DataFrame
            DataFrame with features parameters

        Example
        -------
        """
        self.x = self.df[self.df.columns[start:end]]
        self.columns = self.x.columns
        return self.x

    def set_labels(self, start, end):
        """Select the labels from the input DataFrame.

        This methods takes the DataFrame and selects all the columns from "start"
        to "end" values to indicate which columns should be treated as labels.

        Parameters
        ----------
        start : int
            Start column of dataframe labels
        end : int
            End column of dataframe labels

        Returns
        -------
        y : pd.DataFrame
            DataFrame with labels parameters

        Example
        -------
        """
        self.y = self.df[self.df.columns[start:end]]
        return self.y

    def feature_reduction(self, n):
        """Feature reduction using Decision Tree Regression method.

        This function uses Decision Tree Regression to select the "n" best features.
        Due it's random aspects, the selected best features may not be the same every
        time this function is called.

        Parameters
        ----------
        n : int
            Number of relevant features.

        Returns
        -------
        x : pd.DataFrame
            Minimum number of features that satisfies "n" for each label.
        """
        # define the model
        model = DecisionTreeRegressor()

        # fit the model
        model.fit(self.x, self.y)

        # get importance
        importance = model.feature_importances_

        # summarize feature importance
        featureScores = pd.concat(
            [pd.DataFrame(self.x.columns), pd.DataFrame(importance)], axis=1
        )
        featureScores.columns = ["Specs", "Score"]
        self.best = featureScores.nlargest(n, "Score")["Specs"].values
        self.x = self.x[self.best]

        return self.x

    def data_scaling(self, test_size, scaling=False, scalers=None):
        """Perform data scalling.

        This function scales the input and output data. The DataFrame provided may have
        multiple variables with different units and leading to completely distinguished
        magnitude values. Unscaled input variables can result in a slow or unstable
        learning process, whereas unscaled target variables on regression problems
        can result in exploding gradients causing the learning process to fail.

        Parameters
        ----------
        test_size : float
            Percentage of data destined for testing.
        scaling : boolean, optional
            Choose between scaling the data or not.
            The default is False.
        scalers : scikit-learn object
            scikit-learn scalers method. Check sklearn.preprocessing for more
            informations about each scaler method.
            The default is None.

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
            self.x, self.y, test_size=test_size
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
            self.x_train = self.x_train.values
            self.x_test = self.x_test.values
            self.y_train = self.y_train.values
            self.y_test = self.y_test.values
            self.scaler1 = None
            self.scaler2 = None

        return self.x_train, self.x_test, self.y_train, self.y_test

    def build_Sequential_ANN(self, hidden, neurons, dropout_layers=None, dropout=None):
        """Construct a sequential Artificial Neural Network from Keras models.

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
        self.model.add(Dense(len(self.x.columns), activation="relu"))
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
        """Run the neural network model.

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
        if self.scaler2 is None:
            self.train = pd.DataFrame(
                self.scaler2.inverse_transform(self.predictions), columns=self.y.columns
            )
            self.test = pd.DataFrame(
                self.scaler2.inverse_transform(self.y_test), columns=self.y.columns
            )
        else:
            self.train = pd.DataFrame(self.predictions, columns=self.y.columns)
            self.test = pd.DataFrame(self.y_test, columns=self.y.columns)

        return self.model, self.predictions

    def model_history(self):
        """Plot model history.

        Plots the history of loss and val_loss vs the number of epochs.

        Returns
        -------
        fig : Plotly.figure
            Loss and Val_Loss data through the epochs.
        """
        path = Path(__file__).parent

        hist = pd.DataFrame(self.history.history)
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
            title=dict(text="Epoch", font=dict(size=15)),
            **axes_default,
        )
        fig.update_yaxes(
            title=dict(text="Log<sub>10</sub>Loss", font=dict(size=15)),
            **axes_default,
        )
        fig.update_layout(
            plot_bgcolor="white",
            legend=dict(
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
            ),
        )
        fig.write_html(str(path / f"models/{self.name}/img/history.html"))
        fig.write_image(str(path / f"models/{self.name}/img/history.png"))

        return fig

    def metrics(self, save=False):
        """Print model metrics.

        This function displays the model metrics after the neural network is being
        built.

        Parameters
        ----------
        save : bool, optional
            Key to decide if the table values should be saved in a HTML table or not.
            True saves the table; False do not.

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
            / (len(self.predictions) - (1 + len(self.x.columns)))
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
        if save:
            HTML_formater(metrics, self.name, "Metrics")

        print(
            "Scores:\nMAE: {}\nMSE: {}\nR^2:{}\nR^2 adjusted:{}\nExplained variance:{}".format(
                MAE, MSE, R2, R2_a, explained_variance
            )
        )

    def hypothesis_test(self, kind="ks", p_value=0.05, save=False):
        """Run a hypothesis test.

        This function runs a hypothesis test based on the observed data modeled as the
        realised values taken by a collection of random variables. There are 2 options
        available to run these tests:
            "ks": Komolgorov-Smirnov test
            "w": Welch test

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
        save : boolean
            If True, saves the hypothesis test. If False, the hypothesis_test won't be
            saved.

        References
        ----------
        Larry Wasserman. 2010. All of Statistics: A Concise Course in Statistical
        Inference. Springer Publishing Company, Incorporated.

        Returns
        -------
        p_df : pd.DataFrame
            Hypothesis test results.
            The first column returns the p-value. The second column returns the status,
            which can reject or not the test result. If the calculated p-value is
            greater than "p_value" arg, the result shall be "Not Reject".

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
            if save:
                HTML_formater(p_df, self.name, "KS Test")

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
            if save:
                HTML_formater(p_df, self.name, "Welch Test")

        return p_df

    def validation(self, x, y):
        """Perform model validation.

        Parameters
        ----------
        x : pd.DataFrame
            DataFrame with feature data.
        y : pd.DataFrame
            DataFrame with labels data.

        Returns
        -------
        train : pd.DataFrame
            DataFrame with the train data.
        test : pd.DataFrame
            DataFrame with the test data.
        """
        self.test = y
        if self.scaler1 is not None:
            x_scaled = self.scaler1.transform(x.values.reshape(-1, len(x.columns)))
        if self.scaler2 is not None:
            self.train = pd.DataFrame(
                self.scaler2.inverse_transform(self.model.predict(x_scaled)),
                columns=y.columns,
            )
        else:
            self.train = pd.DataFrame(self.model.predict(x_scaled), columns=y.columns)

        return self.train, self.test

    def postprocessing(self):
        """Create an instance to plot results for neural networks.

        This method returns an instance from PostProcessing class. It allows plotting
        some analyzes for neural networks and a HTML report with a result summary.
            - plot_overall_results
            - plot_confidence_bounds
            - plot_qq
            - plot_standardized_error
            - plot_residuals_resume
            - show

        Returns
        -------
        results : PostProcessing object
            An instance from PostProcessing class that allows plotting some analyzes
            for neural networks.
        """
        results = PostProcessing(self.train, self.test, self.name)
        return results

    def save(self):
        """Save a neural netowork model.

        This function saves the required files for the neural network model.
        It creates a new folder named after the "name" argument passed to Pipeline
        initialization. This folder can be found within the rossml folder package.

        If the model has the same name as a previously saved neural network,
        the old files will be replaced with the files from the new model.
        """
        path = Path(__file__).parent / f"models/{self.name}"
        if not path.exists():
            path.mkdir()
        else:
            for child in path.glob("*"):
                if child.is_file():
                    child.unlink()

        self.model.save(path / r"{}.h5".format(self.name))
        dump(self.y.columns, open(path / r"{}_columns.pkl".format(self.name), "wb"))
        if self.best is not None:
            dump(
                self.best, open(path / r"{}_best_features.pkl".format(self.name), "wb")
            )
        dump(self.columns, open(path / r"{}_features.pkl".format(self.name), "wb"))
        dump(
            self.df.describe(), open(path / r"{}_describe.pkl".format(self.name), "wb")
        )
        if self.scaler1 is not None:
            dump(self.scaler1, open(path / r"{}_scaler1.pkl".format(self.name), "wb"))
        if self.scaler2 is not None:
            dump(self.scaler2, open(path / r"{}_scaler2.pkl".format(self.name), "wb"))


class Model:
    """Retrieve a previously neural network saved.

    This class loads any neural network built from Pipeline class. It can be used to
    retrieve the model and predict new coefficients from a data set.

    Parameters
    ----------
    name : str
        The neural network folder's name to load.

    Returns
    -------
    A neural network model
    """

    def __init__(self, name):
        self.name = name
        self.load()

    def load(self):
        """Load a neural network model from rossml folder."""
        path = Path(__file__).parent / f"models/{self.name}"
        self.model = load_model(path / r"{}.h5".format(self.name))
        self.columns = load(open(path / r"{}_columns.pkl".format(self.name), "rb"))
        self.features = load(open(path / r"{}_features.pkl".format(self.name), "rb"))
        self.describe = load(open(path / r"{}_describe.pkl".format(self.name), "rb"))

        # load best features
        try:
            self.best = load(
                open(path / r"{}_best_features.pkl".format(self.name), "rb")
            )
        except FileNotFoundError:
            self.best = None

        # load the scaler
        try:
            self.scaler1 = load(open(path / r"{}_scaler1.pkl".format(self.name), "rb"))
        except FileNotFoundError:
            self.scaler1 = None

        try:
            self.scaler2 = load(open(path / r"{}_scaler2.pkl".format(self.name), "rb"))
        except FileNotFoundError:
            self.scaler2 = None

    def predict(self, x):
        """Predict rotordynamics coefficients.

        This function takes a Series or a DataFram with input values (must match the
        model features) and uses it to calculate new coefficients.

        Parameters
        ----------
        x : pd.Series, pd.DataFrame
            A series or dataframe with input data to predict new rotordynamics
            coefficients.

        Returns
        -------
        results : pd.DataFrame
            A DataFrame with predictions for the rotordynamics coefficients.
        """
        if self.best is None:
            self.x = x
        else:
            self.x = x[self.best]

        if len(self.x.shape) == 1:
            self.x = self.x.values.reshape(1, -1)

        if self.scaler1 is not None:
            self.x = self.scaler1.transform(self.x)
        else:
            self.x = x

        if self.scaler2 is not None:
            predictions = self.model.predict(self.x)
            self.results = pd.DataFrame(
                self.scaler2.inverse_transform(predictions), columns=self.columns
            )
        else:
            self.results = pd.DataFrame(predictions, columns=self.columns)

        return self.results

    def coefficients(self):
        """Return the predicted stiffness and damping coefficients.

        This function only returns an output if Model.predict() has been run.
        It takes the results from Model.predict() and slices it into K (stiffness) and
        C (damping) lists.

        Returns
        -------
        K : list
            List with the predicted stiffness coefficientes.
        C : list
            List with the predicted damping coefficientes.
        """
        self.K = []
        self.C = []
        for results in self.results.values:
            self.K.append(results[0:4].reshape(2, 2))
            self.C.append(results[4:].reshape(2, 2))

        return self.K, self.C


class PostProcessing:
    """Class used to store and postprocess results from a neural network model.

    Parameters
    ----------
    train : pd.DataFrame
        DataFrame with the train data.
    test : pd.DataFrame
        DataFrame with the test data.
    name : str
        The neural network folder's name to load.
    """

    def __init__(self, train, test, name):
        self.train = train
        self.test = test
        self.test.index = self.train.index
        self.name = name

    def plot_overall_results(self, save_fig=False):
        """Create a pairplot with train and test data.

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
        path = Path(__file__).parent
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
            df,
            diag="box",
            index="label",
            title="",
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
            fig.write_html(str(path / f"models/{self.name}/img/pairplot.html"))
            fig.write_image(str(path / f"models/{self.name}/img/pairplot.png"))

        return fig

    def plot_confidence_bounds(self, a=0.01, percentile=0.05, save_fig=False):
        """Plot and save a confidence interval based on DKW inequality.

        Parameters
        ----------
        a : float
            Significance level. A number between 0 and 1.
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
        path = Path(__file__).parent
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
                    name=f"{v} - {100 * (1 - a)}% Confidence Interval",
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
                    name=f"train {v}",
                    legendgroup=f"train {v}",
                    hoverinfo="none",
                )
            )

            fig.update_xaxes(
                title=dict(text=f"{v}", font=dict(size=15)),
                **axes_default,
            )
            fig.update_yaxes(
                title=dict(text="Cumulative Density Function", font=dict(size=15)),
                **axes_default,
            )
            fig.update_layout(
                plot_bgcolor="white",
                legend=dict(
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                ),
            )
            figures.append(fig)

            if save_fig:
                fig.write_html(str(path / f"models/{self.name}/img/CI_{v}.html"))
                fig.write_image(str(path / f"models/{self.name}/img/CI_{v}.png"))
        return figures

    def plot_qq(self, save_fig=False):
        """Plot and save a qq-plot of the training set.

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
        path = Path(__file__).parent
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
                legend=dict(
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                ),
            )
            figures.append(fig)

            if save_fig:
                fig.write_html(str(path / f"models/{self.name}/img/qq_plot_{v}.html"))
                fig.write_image(str(path / f"models/{self.name}/img/qq_plot_{v}.png"))

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
        path = Path(__file__).parent
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
            error = self.test[v] - self.train[v]
            error = error / error.std()
            error.dropna(inplace=True)
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=self.test[v],
                    y=error,
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
                title=dict(text=f"Standardized Residuals", font=dict(size=15)),
                **axes_default,
            )
            fig.update_layout(
                title_text=f"Standard Deviation: {np.round(np.std(error), 2)}",
                plot_bgcolor="white",
                legend=dict(
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                ),
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
                fig.write_html(
                    str(
                        path
                        / f"models/{self.name}/img/standardized_error_{v}_plot.html"
                    )
                )
                fig.write_image(
                    str(
                        path / f"models/{self.name}/img/standardized_error_{v}_plot.png"
                    )
                )

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
        path = Path(__file__).parent
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
            fig.write_html(str(path / f"models/{self.name}/img/residuals_resume.html"))
            fig.write_image(str(path / f"models/{self.name}/img/residuals_resume.png"))

        return fig

    def report(self, file="results", **kwargs):
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
        _ = self.plot_confidence_bounds(save_fig=True, **kwargs)
        _ = self.plot_qq(save_fig=True)
        _ = self.plot_standardized_error(save_fig=True)
        _ = self.plot_residuals_resume(save_fig=True)

        from_path = Path(__file__).parent / "template"
        to_path = Path(__file__).parent / f"models/{self.name}"
        for f in from_path.glob("**/*"):
            shutil.copy(f, to_path)
        return webbrowser.open(str(to_path / f"{file}.html"), new=1)
