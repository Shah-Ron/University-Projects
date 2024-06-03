## URL:
The URL for the site can be accessed using https://chchoptimizebusroute.azurewebsites.net/ . Sporadically the data from the API fails to load due to CORS issue even after the CORS settings are applied. Refreshing the page couple of times or accessing after sometime fixes the problem. Still we are yet to determine the root cause of this issue. 

## AIM:

This repo focuses on creating a webpage for plotting the actual bus routes and the construction sites on map.

## PROCEDURE:

This was implemented with the help of HTML and JavaScript. A HTML webpage was created which was integrated with Azure Maps. Inorder to use Azure Maps, we created an Azure Map Azzount in Azure Maps, which will let us use Azure Map API's and Azure Map SDK's. We used these features to call the map and plot our coordinates

As part of achieving what was aimed for, the next step was to call the API's that were created for fetching bus routes and Construction Sites. Once these data were fetched, Javascript functions were written to plot these on Azure Map. Plotting the bus routes were pretty straightforward, as the data was given in a sequential order. The only factor which was hard about fetching bus routes coordinates were just that there was a need to access multiple tables inorder to obtain one single route. But when it came to construction sites, multiple functions had to be done, as these data which was obtained were not correct or even in the required order of plotting. Multiple functions, such as finding the haversine distance from a reference point and sorting it out accordingly, were done. Even with the sorted coordinates, the data were not accurate and hence there was a need to smoothen it out a bit.

Once all these coordinates were obtained, color coded the data points and plotted it on Azure Maps in the HTML webpage. 

## Limitation:

Optimisation was unsuccessful. Tried using A* algorithms inorder to get the points of intersection of these coordinates and tried using Azure Maps Route Service API to avoid these particular polygons that was created with the data points. But the coordinates that was obtained were not accurate and was going all around the places. This requires a greater algorithm and more data as the data we used were almost half the time inaccurate from the beginning. There is a high chance that this failed because the data was inaccurate and less in number from the beginning and some flaw in algorithm. Also one limitation to be noticed,was that Azure Maps Route Service API had restrictions on the number of coordinates that can be send to it, restricting it to 150 coordinates per API call. Bus routes had almost 1200 coordinates per bus route and this forced the algorithm to be written in such a way that a chunk of the coordinate was send first and then repeated in a loop. This gave us not very accurate data points as we observed some data points running amok for simplest of change in bus routes.
