using OptimizeBusRouteAPI.Entities;
using static OptimizeBusRouteAPI.Data.DbContextClass;

namespace OptimizeBusRouteAPI.Repositories
{
    public interface IMapsService
    {
        //public Task<IEnumerable<TripData>> GetTripData(string Id);
        public Task<IEnumerable<dynamic>> GetAllRoutes();
        public Task<IEnumerable<dynamic>> GetAllTripsByRoute(string RouteID);
        public Task<IEnumerable<dynamic>> GetMetroTripRoutes(string RouteID, Int32 TripID);
        public Task<IEnumerable<dynamic>> GetConstructionSites();
    }
}
