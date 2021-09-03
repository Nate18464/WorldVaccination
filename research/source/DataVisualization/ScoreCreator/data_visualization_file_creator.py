"""
This modules extracts and cleans the original vaccination data in order
to have only the date, country, and people_fully_vaccinated_per_hundred columns

Then, it fits a curve and saves the r-squared values

After, it combines these predictions with the original data

Finally, it creates a csv file with the data
"""
import warnings
import pandas as pd
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
    return data_dict

def get_score_dataframe(country, model, model_type):
    """Formats a dataframe with the columns we want

    Creates a dataframe with columns for location,
    model_type, and r-squared value. Each with one
    corresponding value with it

    Args:
        country:
            The country that this r-squared value corresponds to
        model:
            The model trained for a certain split and country data
        model_type:
            A string that holds the type of model

    Returns:
        A DataFrame with a location, model_type, and r_squared.
    """
    new_data = pd.DataFrame()
    new_data["location"] = [country]
    new_data["model_type"] = [model_type]
    new_data["r_squared"] = model.get_score()
    return new_data

def get_split_scores(data_dict, end_split):
    """Extracts the r-squared values for different train-test splits

    Loops through each country and makes new entry for the country
    with the r-squared values for each model for either an end split
    or a random split

    Args:
        data_dict:
            Dictionary that holds key as country and value as data
            for that country
        end_split:
            Boolean that decides whether the split will be random or end

    Returns:
        A dataframe with r-squared values for each model.
    """
    new_data = pd.DataFrame()
    for country in data_dict.keys():
        data = data_dict[country]["data"]
        models = [LogisticRegressionModel(data, end_split)]
        models.append(PolynomialRegressionModel(data, end_split))
        models.append(LogisticLogarithmicRegressionModel(data, end_split))
        models.append(LogisticPolynomialRegressionModel(data, end_split))
        model_types = ["logistic_score", "polynomial_score",
                       "logistic_logarithmic_score",
                       "logistic_polynomial_score"]
        split = "_random_split"
        if end_split:
            split = "_end_split"
        for i in range(len(model_types)):
            model_types[i] += split
        for i in range(len(models)):
            model = models[i]
            model_type = model_types[i]
            new_data = new_data.append(get_score_dataframe(country, model, model_type))

    return new_data

def combine_data(random_data, end_data):
    """Merges two dataframes into one using the location column

    Args:
        random_data:
            Dataframe that has r-squared values with random split
        end_data:
            Dataframe that has r-squared values with end split

    Returns:
        Dataframe with combined data
    """
    return random_data.append(end_data)

def __main__():
    data_dict = extract_data()
    random_data = get_split_scores(data_dict, False)
    end_data = get_split_scores(data_dict, True)
    all_data = combine_data(random_data, end_data)
    all_data.to_csv("../../../resource/DataVisualization/rsquared_data.csv", index=False)

__main__()
