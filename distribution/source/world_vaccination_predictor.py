"""
This program uses the vaccinations.csv found in the resource folder
this file can be updated at
https://github.com/owid/covid-19-data/tree/master/public/data/vaccinations
This program creates a GUI for the user to look at predictions for how % fully vaccinated a country
will be based on a given date.
The prediction is made by transforming the y values into a Logistic curve, then fitting that
to a Linear Regression of a straight line, or a Logarithmic Curve.
"""

import tkinter as tk_gui_library
from tkinter import ttk as ttk_gui_library
import warnings
import datetime
from tkcalendar import Calendar
import pandas as pd
from polynomial_regression import PolynomialRegressionModel
from logistic_regression import LogisticRegressionModel
from logistic_logarithmic_regression import LogisticLogarithmicRegressionModel
from logistic_polynomial_regression import LogisticPolynomialRegressionModel
# Imports needed for PyInstaller
import sklearn.utils._weight_vector
import babel.numbers
warnings.filterwarnings("ignore")

def extract_data():
    """Extract data from vaccinations.csv

    Creates a data frame from vaccinations.csv
    Transforms the data into data that is useable by models
    Then extracts data from each country

    Returns:
        A dictionary where the key is the country and
        the value holds information for later, for now just the data
        and the min_date which is just the minimum of the date column
    """
    try:
        file = open("config.txt", "r")
    except FileNotFoundError:
        print("Config File Missing")
        exit()
    config_commands = file.read().split("\n")
    file_path = config_commands[0].split(":")[1]
    raw_data = pd.read_csv(file_path)
    file.close()
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
    return data_dict, min_date

def create_frames(frames):
    """Create initial frame structure

    Create a root and mainframe
    Mainframe will be used to hold all of the widgets
tk
    Args:
        frames:
            An empty dictionary that we add the root
            and mainframe to
    """
    frames["root"] = tk_gui_library.Tk()
    frames["root"].title("% Fully Vaccinated Predictor")

    frames["mainframe"] = ttk_gui_library.Frame(frames["root"], padding="3 3 12 12")
    frames["mainframe"].grid(column=0, row=0, sticky=(tk_gui_library.N,
                                                      tk_gui_library.W,
                                                      tk_gui_library.E,
                                                      tk_gui_library.S))

def create_options(dependencies):
    """Create the options to be displayed in the listbox

    Creates a list of countries to be displayed in the listbox

    Args:
        dependencies:
            A dictionary holding all the dependencies
    """
    dependencies["options"] = sorted(dependencies["data_dict"].keys())

def create_labels(mainframe, widgets, dependencies):
    """Creates 3 labels

    The first label is a label to show the selected country
    The second label is the actual, which holds what the actual
    people_fully_vaccinated_per_hundred value
    The last label is the predicted that holds the predicted
    people_fully_vaccinated_per_hundred value predicted by our model

    Args:
        mainframe:
            A frame named mainframe that holds all the widgets
        labels:
            A dictionary that holds all the label widgets
    """
    labels = dict()
    labels["country"] = ttk_gui_library.Label(mainframe, text="Selected Country: "+
                                             dependencies["options"][0])
    labels["country"].grid(column=3, row=0)

    labels["actual"] = ttk_gui_library.Label(mainframe, text="Actual")
    labels["actual"].grid(column=3, row=3)

    labels["predicted"] = ttk_gui_library.Label(mainframe, text="Predicted")
    labels["predicted"].grid(column=3, row=4)

    widgets["labels"] = labels

def create_listbox(mainframe, widgets, options):
    """Create a listbox with all the countries

    Creates a listbox and scroll bar to go through all the countries
    Adds an element "listbox" to the widgets

    Args:
        mainframe:
            A frame named mainframe that holds all the widgets
        widgets:
            A dictionary that holds all the widgets
        options:
            A list of options that holds all of the possible countries
    """
    listbox = tk_gui_library.Listbox(mainframe)
    listbox.grid(row=0, column=0, rowspan=5)
    scrollbar = tk_gui_library.Scrollbar(mainframe)
    scrollbar.grid(row=0, column=1, rowspan=5)

    for values in options:
        listbox.insert(tk_gui_library.END, values)

    listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)
    widgets["listbox"] = listbox
    widgets["scrollbar"] = scrollbar

def create_model_selector(mainframe, widgets, dependencies):
    """Create a listbox with all the countries

    Creates a listbox and scroll bar to go through all the countries
    Adds an element "listbox" to the widgets

    Args:
        mainframe:
            A frame named mainframe that holds all the widgets
        widgets:
            A dictionary that holds all the widgets
        options:
            A dictionary holding all the dependencies
    """
    listbox = tk_gui_library.Listbox(mainframe)
    listbox.grid(row=10, column=0, rowspan=4)
    options = ["Polynomial", "Logistic", "Logistic Logarithmic", "Logistic Polynomial"]
    for values in options:
        listbox.insert(tk_gui_library.END, values)
    widgets["model_selector"] = listbox
    widgets["model"] = ttk_gui_library.Label(mainframe, text="Selected Model: Logistic Logarithmic")
    widgets["model"].grid(column=3, row=2)
    dependencies["model"] = "Logistic Logarithmic"

def create_cal(mainframe, widgets):
    """Creates a calender to pick dates

            A frame named mainframe that holds all the widgets
        widgets:
            A dictionary holding all widgets
    """
    today = datetime.date.today() + datetime.timedelta(days=14)

    widgets["cal"] = Calendar(mainframe, selectmode='day',
                              year=today.year, month=today.month,
                              day=today.day)
    widgets["cal"].grid(column=0, row=5, columnspan=2, rowspan=5)

def create_date(mainframe, widgets):
    """Creates a label with the selected date

    Creates a button that saves the date selected by the calendar
    Also creates a label to display the last selected date

    Args:
        mainframe:
            A frame named mainframe that holds all the widgets
        widgets:
            A dictionary holding all widgets
    """
    widgets["date"] = ttk_gui_library.Label(mainframe, text="Selected Date: "+
                                            widgets["cal"].get_date())
    widgets["date"].grid(column=3, row=1)

def pad_widgets(mainframe, date):
    """Add padding to every element

    Loops through each child of mainframe and adds padding

    Args:
        mainframe:
            A frame named mainframe that holds all the widgets
        date:
            A widget that holds the date for the predicted value
    """
    for child in mainframe.winfo_children():
        child.grid_configure(padx=5, pady=5, sticky=(tk_gui_library.N,
                                                     tk_gui_library.W,
                                                     tk_gui_library.S,
                                                     tk_gui_library.E))
    date.grid_configure(sticky=(tk_gui_library.N,
                                tk_gui_library.W,
                                tk_gui_library.S))

def update_selected_country(widgets, dependencies):
    """Updates the selected country

    Updates the selected_country in dependencies
    And updates the label on the screen displaying the
    selected country

    Args:
        widgets:
            A dictionary holding all of the widgets
        dependencies:
            A dictionary holding all of the dependencies
    """
    for i in widgets["listbox"].curselection():
        dependencies["selected_country"] = widgets["listbox"].get(i)
    widgets["labels"]["country"].config(text="Selected Country: "+
                                        dependencies["selected_country"])

def update_actual(widgets, dependencies):
    """A function that updates the actual data

    This function displays the most recent entry in the data
    Displaying the date and people_fully_vaccinated_per_hundred

    Args:
        widgets:
            A dictionary holding all of the widgets
        dependencies:
            A dictionary holding all of the dependencies
    """
    data_dict = dependencies["data_dict"]
    selected_country = dependencies["selected_country"]
    min_date = dependencies["min_date"]
    date = data_dict[selected_country]["data"]["date"].max()
    date = datetime.timedelta(days=int(date)) + min_date
    actual = data_dict[selected_country]["data"]["people_fully_vaccinated_per_hundred"].max()
    widgets["labels"]["actual"].config(text="As of "+date.strftime("%m/%d/%y")+
                                       ", % Fully Vaccinated: "+str(actual*100)+"%")

def find_close(data, date):
    """Finds the closest datapoint to a given date

    Loops through days away from the given date until eventually finding
    a data point that is in the dataframe

    Args:
        data:
            Dataframe with the data for the certain country selected
        date:
            An integer corresponding to the date to look for the closest
            data point from

    Returns:
        A datapoint corresponding to people_fully_vaccinated_per_hundred
        for the datapoint closest to the given date and the date for the
        datapoint closest to the given date.
    """
    add_days = 0
    while True:
        add = data.loc[data.date == date+add_days]
        sub = data.loc[data.date == date-add_days]
        if len(add) != 0 or len(sub) != 0:
            if len(add) != 0:
                return add["people_fully_vaccinated_per_hundred"].iloc[0], add["date"].iloc[0]
            return sub["people_fully_vaccinated_per_hundred"].iloc[0], sub["date"].iloc[0]
        add_days += 1

def predict(widgets, dependencies):
    """A function that makes and displays a prediction

    This function makes a prediction based on our RegressionModel
    Class, then displays it to the screen.

    Args:
        widgets:
            A dictionary holding all of the widgets
        dependencies:
            A dictionary holding all of the dependencies
    """
    selected_country = dependencies["selected_country"]
    data = dependencies["data_dict"][selected_country]
    min_date = dependencies["min_date"]
    date = widgets["cal"].get_date()
    date = datetime.datetime.strptime(date, "%m/%d/%y")
    date = (date-min_date).days
    if date <= data["data"]["date"].max():
        closest_percentage, closest_date = find_close(data["data"], date)
        closest_date = datetime.timedelta(days=int(closest_date)) + min_date
        widgets["labels"]["predicted"].config(text="Closest % Fully Vaccinated: "
                                              +str(closest_percentage*100)+"% on "
                                              +closest_date.strftime("%m/%d/%Y"))
    else:
        if dependencies["model"] == "Polynomial":
            model = PolynomialRegressionModel(data["data"])
        elif dependencies["model"] == "Logistic":
            model = LogisticRegressionModel(data["data"])
        elif dependencies["model"] == "Logistic Logarithmic":
            model = LogisticLogarithmicRegressionModel(data["data"])
        elif dependencies["model"] == "Logistic Polynomial":
            model = LogisticPolynomialRegressionModel(data["data"])
        else:
            print("Error")
        predicted = model.predict([[date]])[0]
        if predicted < data["data"]["people_fully_vaccinated_per_hundred"].max():
            predicted = data["data"]["people_fully_vaccinated_per_hundred"].max()
        predicted = round(predicted*100, 3)
        widgets["labels"]["predicted"].config(text="Predicted % Fully Vaccinated: "
                                              +str(predicted)+"%")

def __main__():
    def update():
        update_actual(widgets, dependencies)
        predict(widgets, dependencies)

    def update_country(event):
        update_selected_country(widgets, dependencies)
        update()

    def update_model(event):
        for i in widgets["model_selector"].curselection():
            dependencies["model"] = widgets["model_selector"].get(i)
        widgets["model"].config(text="Selected Model: "+
                                dependencies["model"])
        update()

    def update_date(event):
        widgets["date"].config(text="Selected Date: "+
                               widgets["cal"].get_date())
        update()

    dependencies = dict()
    dependencies["data_dict"], dependencies["min_date"] = extract_data()

    frames = dict()
    create_frames(frames)
    create_options(dependencies)

    widgets = dict()
    create_labels(frames["mainframe"], widgets, dependencies)

    # Set the default country to the first in options
    dependencies["selected_country"] = dependencies["options"][0]

    create_listbox(frames["mainframe"], widgets, dependencies["options"])
    create_model_selector(frames["mainframe"], widgets, dependencies)
    create_cal(frames["mainframe"], widgets)

    create_date(frames["mainframe"], widgets)
    update()
    pad_widgets(frames["mainframe"], widgets["date"])
    widgets["listbox"].bind("<<ListboxSelect>>", update_country)
    widgets["model_selector"].bind("<<ListboxSelect>>", update_model)
    widgets["cal"].bind("<<CalendarSelected>>", update_date)
    frames["root"].mainloop()

__main__()
