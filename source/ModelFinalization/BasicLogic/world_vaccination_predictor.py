"""
This program uses the vaccinations.csv found in the resource folder
this file can be updated at
https://github.com/owid/covid-19-data/tree/master/public/data/vaccinations
This program creates a GUI for the user to look at predictions for how % fully vaccinated a country
will be based on a given date.
The prediction is made by transforming the y values into a Logistic curve, then fitting that
to a Linear Regression of a straight line, or a Logarithmic Curve.
"""

import tkinter as tk
from tkinter import ttk
import warnings
import datetime
from tkcalendar import Calendar
import pandas as pd
from logistic_logarithmic_regression import LogisticLogarithmicRegressionModel
# Imports needed for PyInstaller
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
    raw_data = pd.read_csv("../../../resource/ModelFinalization/vaccinations.csv")
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

    Args:
        frames:
            An empty dictionary that we add the root
            and mainframe to
    """
    frames["root"] = tk.Tk()
    frames["root"].title("% Fully Vaccinated Predictor")

    frames["mainframe"] = ttk.Frame(frames["root"], padding="3 3 12 12")
    frames["mainframe"].grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

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
    labels["country"] = ttk.Label(mainframe, text="Selected Country: "+
                                  dependencies["options"][0])
    labels["country"].grid(column=3, row=0)

    labels["actual"] = ttk.Label(mainframe, text="Actual")
    labels["actual"].grid(column=3, row=2)

    labels["predicted"] = ttk.Label(mainframe, text="Predicted")
    labels["predicted"].grid(column=3, row=3)

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
    listbox = tk.Listbox(mainframe)
    listbox.grid(row=0, column=0, rowspan=5)
    scrollbar = tk.Scrollbar(mainframe)
    scrollbar.grid(row=0, column=1, rowspan=5)

    for values in options:
        listbox.insert(tk.END, values)

    listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)
    widgets["listbox"] = listbox
    widgets["scrollbar"] = scrollbar

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
    widgets["date"] = ttk.Label(mainframe, text="Selected Date: "+
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
        child.grid_configure(padx=5, pady=5, sticky=(tk.N, tk.W, tk.S, tk.E))
    date.grid_configure(sticky=(tk.N, tk.W, tk.S))

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
    data_dict = dependencies["data_dict"]
    min_date = dependencies["min_date"]
    selected_country = dependencies["selected_country"]
    date = widgets["cal"].get_date()
    date = datetime.datetime.strptime(date, "%m/%d/%y")
    date = (date-min_date).days
    predicted = 0
    try:
        predicted = data_dict[selected_country]["model"].predict([[date]])
    except KeyError:
        data_dict[selected_country]["model"] = LogisticLogarithmicRegressionModel(
            data_dict[selected_country]["data"])
        predicted = data_dict[selected_country]["model"].predict([[date]])
    widgets["labels"]["predicted"].config(text="Predicted % Fully Vaccinated: "
                                          +str(predicted[0]*100)+"%")

def __main__():
    def update():
        update_selected_country(widgets, dependencies)
        update_actual(widgets, dependencies)
        predict(widgets, dependencies)

    def update_country(event):
        update_selected_country(widgets, dependencies)
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
    create_cal(frames["mainframe"], widgets)

    create_date(frames["mainframe"], widgets)
    update()
    pad_widgets(frames["mainframe"], widgets["date"])
    widgets["listbox"].bind("<<ListboxSelect>>", update_country)
    widgets["cal"].bind("<<CalendarSelected>>", update_date)
    frames["root"].mainloop()

__main__()
