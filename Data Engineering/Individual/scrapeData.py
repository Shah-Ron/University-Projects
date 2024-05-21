import requests
from bs4 import BeautifulSoup
import os

def get_first_link():

    # Getting the URL for the latest data from the Prison Statistics URL

    url = "https://www.corrections.govt.nz/resources/statistics/quarterly_prison_statistics"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    div = soup.find("div", {"class": "test"})

    # This will provide the URL for the first link in the Prison Statistics page
    first_link = div.find("a")["href"]
    
    # Now we get the download link from this URL
    response = requests.get(first_link)
    soup = BeautifulSoup(response.content, "html.parser")
    div = soup.find("div", id = lambda x: x and 'component' in x)
    download_link = div.find("a")["href"]

    # Now we get th data we are looking for
    filename = os.path.basename(download_link)
    print("Downloading:", filename)
    # Send a GET request to the URL
    response = requests.get(download_link)
    cwd_directory = os.getcwd()
    directory = r"Download"
    directory = os.path.join(cwd_directory,directory)
    print(directory)
    full_path = os.path.join(directory,filename)
    if response.status_code == 200:
            # Write the content of the response to a file
            with open(full_path, 'wb') as f:
                f.write(response.content)
            print("File downloaded successfully.")
    else:
        print("Failed to download file. Status code:", response.status_code)
    directory = os.path.join(directory, "filename.txt")
    with open(directory,'w') as f:
         f.write(filename)

get_first_link()