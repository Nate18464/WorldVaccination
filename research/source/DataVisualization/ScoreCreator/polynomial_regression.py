"""
This module impliments a class called PolynomialRegressionModel
It creates linear transformations on the x values in order to
have non-linearity. The module uses sklearn's LinearRegression
in order to fit to a line.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def transform_x(x_data, degree):
    """Transforms x values based on the bool passed in

    Either leaves the X value the same for a Linear Model
    Or adds a new row that has logarithmic values

    Args:
        x_data: the x values to be transformed
        log_bool: a boolean to say whether to use a

    Returns:
        returns the transformed x values
    """
    new_x = []
    # Loop through each row in the list
    for row in x_data:
        new_values = []
        # Loop through each possible transformation
        for val in row:
            for exponent in range(2, degree+1):
                new_values.append(val**exponent)
        new_x.append(np.append(row, new_values))
    return new_x

class PolynomialRegressionModel:
    """This Class is a wrapper in order to fit and predict a model

    The Class accepts a dataframe when initialized, then splits that into x and y components
    From there, it tests whether a linear or logarithmic model would be better
    The model overall, fits to a logistic curve by transforming the y values
    Then, this class can make predictions based on x values

    Attributes:
        x_data: holds the x component of the data
        y_data: holds the y component of the data
        model: holds the trained model
        score: holds the r-squared value from the model
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
        self.find_regress()


    def find_regress(self):
        """Checks whether a logarithmic curve or linear curve is better

        Creates a model for a linear and logarithmic curve and compares
        the r-squared values to decide which model to use
        This function sets the model, score, and bool attributes
        """
        self.model, self.score = self.fit(1)
        self.degree = 1
        # Loop through all possible boolean values
        for degree in range(2, 8):
            # Run the fit function
            tmpmodel, tmpscore = self.fit(degree)
            # Check if this new model has a better score
            if tmpscore > self.score:
                # If the new model has a better score, update the model, score, and regress_bool
                self.model = tmpmodel
                self.score = tmpscore
                self.degree = degree

    def fit(self, degree):
        """Transforms the x and y values, then fits the model, returning the model and score

        Calls transform_x and transform_y_fit in order to transform x and y values
        The function then fits a model using the sklearn LinearRegression

        Args:
            bool: a boolean value that says whether to use a
                  logarithmic curve, is passed to transform_x

        Returns:
            A fit model and the r-squared value from the curve
        """
        train_x = transform_x(self.train_x, degree)
        test_x = transform_x(self.test_x, degree)
        model = LinearRegression().fit(train_x, self.train_y)
        score = model.score(test_x, self.test_y)
        return model, score

    def get_score(self):
        """Returns r-squared value from testing

        Returns:
            An r-squared value from testing
        """
        return self.score
