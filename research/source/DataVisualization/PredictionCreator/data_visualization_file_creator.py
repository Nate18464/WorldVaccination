"""
This modules extracts and cleans the original vaccination data in order
to have only the date, country, and people_fully_vaccinated_per_hundred columns

Then, it makes predictions and saves them into a pandas dataframe

After, it combines these predictions with the original data

Finally, it creates a csv file with the data
"""
import warnings
import datetime
import pandas as pd
import numpy as np
from logistic_logarithmic_regression import LogisticLogarithmicRegressionModel
from logistic_regression import LogisticRegressionModel
from logistic_polynomial_regression import LogisticPolynomialRegressionModel
from polynomial_regression import PolynomialRegressionModel
warnings.filterwarnings("ignore")

def extract_data():
    """Extract data from vaccinations.csv

    Creates a data frame from vaccinations.csv
    Transforms the data into data that is useable by models
    Then extracts data from each country

    Returns:
        A dictionary where the key is the country and
        the value holds information for later, for now just the data
    """
    raw_data = pd.read_csv("../../../resource/DataVisualization/vaccinations.csv")
    raw_data = raw_data[["location", "date", "people_fully_vaccinated_per_hundred"]]
    raw_data.date = pd.to_datetime(raw_data.date, format="%Y-%m-%d")
    min_date = raw_data.date.min()
    raw_data.date = raw_data.date-min_date
    raw_data.date = pd.Series([x.days for x in raw_data.date])
    raw_data.drop(raw_data.loc[raw_data.people_fully_vaccinated_per_hundred.isnull()].index,
                  axis=0, inplace=True)
    raw_data["people_fully_vaccinated_per_hundred"] /= 100

    data_dict = dict()
    for country in raw_data.location.unique():
        if len(raw_data.loc[raw_data.location == country]) >= 100:
            tmp_data = raw_data.loc[raw_data.location == country]
            tmp_data.drop("location", axis=1, inplace=True)
            data_dict[country] = {"data":tmp_data}
        else:
            raw_data.drop(raw_data.loc[raw_data.location ==
                                       country].index, inplace=True)
    return data_dict, min_date, raw_data

def make_predictions(data_dict):
    """Make predictions for all models

    Creates a new dataframe that holds the prediction
    for dates from 0 to 499 days from the first entry

    Args:
        data_dict:
            Dictionary that holds the data for countries
            the keys are country and the value is the
            vaccination data

    Returns:
        new_data:
            A dataframe that holds the date, country, and predictions
    """
    new_data = pd.DataFrame(columns=["location", "date",
                                     "logistic_prediction",
                                     "logistic_logarithmic_prediction",
                                     "logistic_polynomial_prediction"])
    for country in data_dict.keys():
        data = data_dict[country]["data"]
        x_data = np.array(list(range(500)))
        model1 = LogisticRegressionModel(data_dict[country]["data"])
        y_pred1 = model1.predict(x_data.reshape(-1, 1))
        model2 = LogisticLogarithmicRegressionModel(data)
        y_pred2 = model2.predict(x_data.reshape(-1, 1))
        model3 = LogisticPolynomialRegressionModel(data)
        y_pred3 = model3.predict(x_data.reshape(-1, 1))
        model4 = PolynomialRegressionModel(data)
        y_pred4 = model4.predict(x_data.reshape(-1, 1))
        data = np.swapaxes(np.array([x_data, y_pred1, y_pred2,
                                     y_pred3, y_pred4]), 0, 1)
        tmp_data = pd.DataFrame(data, columns=["date",
                                               "logistic_prediction",
                                               "logistic_logarithmic_prediction",
                                               "logistic_polynomial_prediction",
                                               "polynomial_prediction"])
        tmp_data["location"] = country
        new_data = new_data.append(tmp_data)
    return new_data

def combine(new_data, raw_data):
    """Combines raw_data with new_data

    Sets the index of each dataframe to location and date
    Then joins the two dataframes

    Args:
        new_data:
            Dataframe holding the predicted values
        raw_data:
            Dataframe holding the original values

    Returns:
        all_data:
            Dataframe with predicted and original values
    """
    return pd.merge(new_data, raw_data, on=["location", "date"], how="outer")

def reformat_date(all_data, min_date):
    """Reformats date to original

    Switches date column from integers of days since first entry
    into a timedelta. Then, adds that timedelta to the min_date
    In the end getting a datetime.

    Args:
        all_data:
            A dataframe holding the original raw data and
            the data from predictive models
    """
    all_data["date"] = [datetime.timedelta(x) for x in all_data["date"]]
    all_data["date"] = all_data["date"] + min_date

def __main__():
    data_dict, min_date, raw_data = extract_data()
    new_data = make_predictions(data_dict)
    all_data = combine(new_data, raw_data)
    reformat_date(all_data, min_date)
    all_data.to_csv("../../../resource/DataVisualization/prediction_data.csv", index=False)

__main__()
