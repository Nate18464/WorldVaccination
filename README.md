# World Vaccination
Both distribution and research have source files in the source directory and resource files in the resource directory

research:
  1. Data Cleaning: This requires to be run on a linux device as it uses linux commands to pull from a github repository for updated vaccination data. If you do not want to pull off of the github, then simply pull the data from this repository or download the vaccines.csv from https://github.com/owid/covid-19-data/tree/master/public/data/vaccinations, and comment out the linux commands in the beginning cells. There is also a version that cleans a larger data set, this can also be downloaded from here or download the owid-covid-data.csv from https://github.com/owid/covid-19-data/tree/master/public/data.
  2. Linear Model Creation: Can be run on any device as long as you run it on jupyter notebook. This section creates and tests a linear regression with different degrees of polynomials in order to see how it would perform. The program also tests different features.
  3. Logistic Model Creation: Can be run on any device as long as you run it on jupyter notebook. This section creates and tests a linear regression fit to a logistic curve with y value transformations. This transforms x values in order to fit to a line, polynomial, and logistic fit. The program also tests different features.
  4. Data Visualization: Is created in Tableau. There are 2 dashboards and a story board. For the dashboards, there is one that let's you look at data from any given country (named Map Selctor) and compare, or one that allows you to select multiple countries to compare data between countries (named Data Comparer). There is also a story board that creates Color Coded Heat maps on top of a map of the world to compare certain statistics in a map visualization. There is also a tableau file that compares the curves of different types of line fits along with the corresponding r-squared values.
  5. Model Finalization: Contains multiple iterations of GUI for displaying predictions with different features for each.
  
distribution:
  Has different python files necessary in order to create a GUI that provides predictions on precentage of people fully vaccinatedbased on lines of best fit along with country. Also contains a shell script that creates a PyInstallere executable.
