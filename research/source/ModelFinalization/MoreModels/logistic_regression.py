"""
This module impliments a class called LogisticRegressionModel
It creates transformations on the y values in order to
fit the data to a logistic curve. The module uses sklearn's LinearRegression
in order to fit to a line.
"""
import math
from sklearn.linear_model import LinearRegression

def transform_y_fit(y_data):
    """Transforms the Y values into a logistic form

    Creates a lambda in order to transform all of the y values
    Uses a list comprehension to loop through all the y values and transform them

    Args:
        y_data: the people_fully_vaccinated_per_hundred column for the country

    Returns:
        A list that holds the transformed y values
    """
    # Create a lambda with formula to transform values.
    transformation = lambda y: -1*math.log((1/(y+.01))-1)
    return [transformation(y) if y < .99 else 10 for y in y_data]

def transform_y_predict(y_data):
    """Transforms the Y values into the non-logistic form

    Creates a lambda in order to transform all of the y values
    Uses a list comprehension to loop through all the y values and transform them

    Args:
        y_data: the people_fully_vaccinated_per_hundred column for the country

    Returns:
        A list that holds the transformed y values
    """
    transformation = lambda y: (1/(1+math.exp(-1*y)))-.01
    return [transformation(y) if y > -70 else 0 for y in y_data]

class LogisticRegressionModel:
    """This Class is a wrapper in order to fit and predict a model

    The Class accepts a dataframe when initialized, then splits that into x and y components
    From there, it tests whether a linear or logarithmic model would be better
    The model overall, fits to a logistic curve by transforming the y values
    Then, this class can make predictions based on x values

    Attributes:
        x_data: holds the x component of the data
        y_data: holds the y component of the data
        model: holds the trained model
        bool: holds a bool for whether or not to use a logarithmic transformation
    """

    def __init__(self, dataframe):
        """Initialize data with X and y portions and call find_regress

        Extracts the x and y values from an inputed dataframe
        This function sets the x_data and y_data attributes

        Args:
            dataframe: a dataframe with a date and people_fully_vaccinated_per_hundred column
        """
        self.x_data = dataframe["date"].to_numpy().reshape(-1, 1)
        self.y_data = dataframe["people_fully_vaccinated_per_hundred"]
        self.fit()

    def fit(self):
        """Transforms the x and y values, then fits the model, returning the model and score

        Calls transform_x and transform_y_fit in order to transform x and y values
        The function then fits a model using the sklearn LinearRegression
        """
        y_data = transform_y_fit(self.y_data.copy())
        self.model = LinearRegression().fit(self.x_data, y_data)

    def predict(self, x_data):
        """Creates a prediction based on x values

        Transforms the X value to make a prediction
        Then transforms the y value into the non-logistic version

        Args:
            x_data: A list of the dates to make a prediction on

        Returns:
            A list of people_fully_vaccinated_per_hundred values that the model predicts
        """
        return transform_y_predict(self.model.predict(x_data))
