This is the individual project I submitted for the client that was brought to us for our data engineering class.

### AIM 

The aim was to collect data from one of the websites present to us, and clean the data, make the meta data, and convert the data into SDMX format and provide the data to the client.

### Explaination:

<b>scrapeData.py</b>:  

The aim of this python file is to scrape the data from the website as provided. The scraping of the data will be automated or scheduled Quarterly as required to get the latest data. Once the data is collected the data will be stored as an excel file and the filename will be saved in a txt file as well. 

<b>cleaning.r</b>

The aim of this R file is to clean the acquired data. I used R here as RStudio is a really good IDE for the visualisation and dealing of data. I've cleaned out all the data and all the Not Available fields have been set to zero as well. Once the cleaning was done, I converted them to CSV files and saved them. 

<b>dataFlask.py</b>

The aim of this python file is to create an API for the data and metadata. It is done using Flask.

<b>metaData.json</b>

This file provides the metadata about the tables and their dimensions and respective observations.

<b>requirements.txt</b>

This file contains the different python and r packages that needs to be installed

<b>Download</b>

The downloaded quarterly statistics will show up here and the text file containing the filename will be placed here as well.

<b>Cleaned</b>

This directory consist of all the cleaned csv files that are obtained after running cleaning.r file.
