"""
This module impliments a class called LogisticRegressionModel
It creates transformations on the y values in order to
fit the data to a logistic curve. The module uses sklearn's LinearRegression
in order to fit to a line.
"""
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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

    def __init__(self, dataframe, end_split):
        """Initialize data with X and y portions and call find_regress

        Extracts the x and y values from an inputed dataframe
        Then splits based on boolean value of end_split
        This function sets the train and test attributes

        Args:
            dataframe:
                A dataframe with a date and people_fully_vaccinated_per_hundred column
            end_split:
                A boolean telling the Class whether to split randomly or at the end
        """
        x_data = dataframe["date"].to_numpy().reshape(-1, 1)
        y_data = dataframe["people_fully_vaccinated_per_hundred"]
        if end_split:
            self.train_x = x_data[:int(len(x_data)*.8)]
            self.test_x = x_data[int(len(x_data)*.8):]
            self.train_y = y_data[:int(len(y_data)*.8)]
            self.test_y = y_data[int(len(y_data)*.8):]
        else:
            split = train_test_split(x_data, y_data, test_size=.2)
            self.train_x, self.test_x, self.train_y, self.test_y = split
        self.fit()

    def fit(self):
        """Transforms the x and y values, then fits the model, returning the model and score

        Calls transform_x and transform_y_fit in order to transform x and y values
        The function then fits a model using the sklearn LinearRegression
        """
        train_y = transform_y_fit(self.train_y)
        test_y = transform_y_fit(self.test_y)
        self.model = LinearRegression().fit(self.train_x, train_y)
        self.score = self.model.score(self.test_x, test_y)

    def get_score(self):
        """Returns r-squared value from testing

        Returns:
            An r-squared value from testing
        """
        return self.score
