# Christchurch Bus Route Optimizer API

## Description
The Christchurch Bus Route Optimizer API is designed to facilitate the creation of optimized bus routes in Christchurch, New Zealand. It connects the database with the user interface by providing endpoints to fetch all routes, trips for each route, coordinates for each route and trip, as well as information about construction sites that may affect the routes.

## Technologies Used
- ASP .NET Core
- C#
- ADO .NET

## Installation
1. Clone the repository.
2. Open the solution in Visual Studio.
3. Restore the necessary NuGet packages.
4. Set up your database connection string in the appsettings.json file.
5. Run the database migrations to create the necessary tables.
6. Build the solution.
7. Run the application.

## Usage
To use the API, follow these steps:
1. Ensure the API is running locally or deployed to a server.
2. Use the provided endpoints to fetch data about bus routes, trips, coordinates, and construction sites.
3. Integrate the API endpoints into your user interface or application.

## Endpoints
The API provides the following endpoints:
- **GET /api/GetAllRoutes**: Retrieves all bus routes.
- **GET /api/GetAllTripsByRoute/{RouteID}**: Retrieves all trips for a specific bus route.
- **GET /api/GetTripData/{RouteID,TripID}**: Retrieves coordinates for a specific bus route based on the trip.
- **GET /api/GetConstructionSites**: Retrieves information about construction sites.

## Database Schema
**Tables**
- ConstructionSites
- Routes
- Shapes
- Trips
**Stored Procedures**
  - SpGetAllRoutes
  - SpGetAllTripsByRouteID
  - SpGetTripRoutes
  - SPGetConstructionSites
