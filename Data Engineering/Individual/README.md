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

### Process:

First, the python file scrapeData.py will run which will scrape the quarterly statistics of the prison data and place it in the folder named Download. The same file will also create a text file named filename.txt which will contain the name of the downloaded file. This is done so that when doing the cleaning part in the R, we can upload the quarterly statistics file without any hiccups. The R file, cleaning.r will run next which will clean the entire data and provide it in a presentable manner. After this the API file, which is dataFlaskAPI.py will be activated, which will give us the required response data in json format. 

All of this has been uploaded into an EC2 instance in AWS where all the files mentioned in the requirements are installed and cron jobs are set such that all of these are rerun every 3 months and get the latest quearterly statistics.

The cron jobs are as below

'''
0 0 1 */3 * ~/university-projects/Data Engineering/Individual/scrapeData.py
5 0 1 */3 * ~/university-projects/Data Engineering/Individual/cleaning.r
6 0 1 */3 * ~/university-projects/Data Engineering/Individual/dataFlaskAPI.py
'''

The cron jobs are set up in such a way that the scraper file will run at 12 am followed by cleaning file which will run 5 mins after the scraper, which is at 12:05 am and then the API will rerun a min after the cleaning function, i.e. that is 12:06 am.
