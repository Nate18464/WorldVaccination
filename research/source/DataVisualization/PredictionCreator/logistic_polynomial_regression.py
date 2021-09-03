"""
This module impliments a class called LogisticLogarithmicRegressionModel
It transforms the y values in order to fit to a logistic curve
It searches for what degreee of polynomia is best
The module uses sklearn's LinearRegression
in order to fit to a line.
"""
from sklearn.linear_model import LinearRegression
from logistic_regression import transform_y_fit
from logistic_regression import transform_y_predict
from polynomial_regression import transform_x

class LogisticPolynomialRegressionModel:
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

    def __init__(self, dataframe):
        """Initialize data with X and y portions and call find_regress

        Extracts the x and y values from an inputed dataframe
        This function sets the x_data and y_data attributes

        Args:
            dataframe: a dataframe with a date and people_fully_vaccinated_per_hundred column
        """
        self.x_data = dataframe["date"].to_numpy().reshape(-1, 1)
        self.y_data = dataframe["people_fully_vaccinated_per_hundred"]
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
        x_data = transform_x(self.x_data, degree)
        y_data = transform_y_fit(self.y_data.copy())
        model = LinearRegression().fit(x_data, y_data)
        score = model.score(x_data, y_data)
        return model, score

    def predict(self, x_data):
        """Creates a prediction based on x values

        Transforms the X value to make a prediction
        Then transforms the y value into the non-logistic version

        Args:
            x_data: A list of the dates to make a prediction on

        Returns:
            A list of people_fully_vaccinated_per_hundred values that the model predicts
        """
        x_data = transform_x(x_data.copy(), self.degree)
        return transform_y_predict(self.model.predict(x_data))
