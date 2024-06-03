using OptimizeBusRouteAPI.Entities;
using OptimizeBusRouteAPI.Repositories;
using Microsoft.AspNetCore.Mvc;
using static OptimizeBusRouteAPI.Data.DbContextClass;
using Newtonsoft.Json;
using System.Text.Json.Serialization;
using System.Text.Json;
using JsonSerializer = System.Text.Json.JsonSerializer;
using System.Data;

namespace OptimizeBusRouteAPI.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class MapsController : Controller
    {

        private readonly IConfiguration _config;
        public MapsController(IConfiguration config)
        {
            _config = config;
        }


        [HttpGet("GetAllRoutes")]
        public async Task<IActionResult> GetAllRoutesAsync()
        {
            // Initialize tripData
            OutData routes = new OutData();

            // Call the service to get trip data
            MapsService service = new MapsService(_config);
            routes = await service.GetAllRoutes();

            var jsonData = routes.ResponseData.Tables[0];
            string jsonResult = JsonConvert.SerializeObject(jsonData, Formatting.Indented);
            return Ok(jsonResult);
        }

        [HttpGet("GetAllTripsByRoute")]
        public async Task<IActionResult> GetAllTripsByRouteAsync(string RouteID)
        {
            // Initialize tripData
            OutData routes = new OutData();

            // Call the service to get trip data
            MapsService service = new MapsService(_config);
            routes = await service.GetAllTripsByRoute(RouteID);

            var jsonData = routes.ResponseData.Tables[0];
            string jsonResult = JsonConvert.SerializeObject(jsonData, Formatting.Indented);
            return Ok(jsonResult);
        }


        [HttpGet("GetTripData")]
        public async Task<IActionResult> GetTripDataAsync(string RouteId, Int32 TripID)
        {
            try
            {
                // Initialize tripData
                OutData tripData = new OutData();

                // Call the service to get trip data
                MapsService service = new MapsService(_config);
                tripData = await service.GetTripData(RouteId, TripID);

                // Check if tripData.ResponseData is null and handle accordingly
                if (tripData.ResponseData == null)
                {
                    return NotFound(); 
                }

                var jsonData = tripData.ResponseData.Tables[0];
                string jsonResult = JsonConvert.SerializeObject(jsonData, Formatting.Indented);
                return Ok(jsonResult);
            }
            catch (Exception ex)
            {
                // Return an error response
                return StatusCode(StatusCodes.Status500InternalServerError, "An error occurred while processing your request.");
            }
        }

        [HttpGet("GetConstructionSites")]
        public async Task<IActionResult> GetConstructionSitesAsync()
        {
            OutData sites = new OutData();
            MapsService service = new MapsService(_config);
            sites = await service.GetConstructionSites();

            var jsonData = sites.ResponseData.Tables[0];
            string jsonResult = JsonConvert.SerializeObject(jsonData, Formatting.Indented);
            return Ok(jsonResult);
        }
    }
}
