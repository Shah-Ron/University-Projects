using OptimizeBusRouteAPI.Data;
using OptimizeBusRouteAPI.Entities;
using Microsoft.Data.SqlClient;
using Microsoft.EntityFrameworkCore;
using System.Data;
using System.Dynamic;
using static OptimizeBusRouteAPI.Data.DbContextClass;

namespace OptimizeBusRouteAPI.Repositories
{
    public class MapsService
    {
        //private readonly DbContextClass _dbContext;

        //public MapsService(DbContextClass dbContext)
        //{
        //    _dbContext = dbContext;
        //}

        private readonly IConfiguration _config;

        public MapsService(IConfiguration config)
        {
            _config = config;
        }

        //public async Task<IEnumerable<TripData>> GetTripData(string Id)
        //{
        //    var param = new SqlParameter("@RouteId", Id);
        //    //return await _dbContext.Trips.FromSqlRaw(@"exec SpGetTripsByID @RouteId", param).ToListAsync();

        //    //var RouteDetails = await Task.Run(() => _dbContext.Trips.FromSqlRaw(@"exec SpGetTripsByID @RouteId", param).ToListAsync());

        //    //var RouteDetails = (IEnumerable<TripData>)await _dbContext.Database.ExecuteSqlRawAsync("EXEC SpGetTripsByID @RouteId", param);
        //    List<TripData> RouteDetails = await Task.Run(() => _dbContext.Trips.FromSqlRaw(@"exec SpGetTripsByID @RouteId", param).ToListAsync());
        //    return RouteDetails;
        //}

        public async Task<OutData> GetAllRoutes()
        {
            try
            {
                OutData dsroutes = new OutData();
                DbContextClass _dbContext = new DbContextClass(_config);

                SqlParameter[] Rparams =
                {
                        new SqlParameter("@RouteID", SqlDbType.VarChar) { Value = "" }
                    };
                dsroutes = await _dbContext.get_Procedure_output_ds("SpGetAllRoutes", Rparams);
                return dsroutes;
            }
            catch (Exception ex)
            {
                throw new Exception(ex.Message);
            }
        }

        public async Task<OutData> GetAllTripsByRoute(string Id)
        {
            try
            {
                OutData dsroutes = new OutData();
                DbContextClass _dbContext = new DbContextClass(_config);

                SqlParameter[] Rparams =
                {
                        new SqlParameter("@RouteID", SqlDbType.VarChar) { Value = Id }
                    };
                dsroutes = await _dbContext.get_Procedure_output_ds("SpGetAllTripsByRouteID", Rparams);
                return dsroutes;
            }
            catch (Exception ex)
            {
                throw new Exception(ex.Message);
            }
        }


        public async Task<OutData> GetTripData(string Id, Int32 TripID)
        {
            try
            {
                OutData dsTripData = new OutData();
                DbContextClass _dbContext = new DbContextClass(_config);

                SqlParameter[] Login_params =
                {
                        new SqlParameter("@RouteID", SqlDbType.VarChar) { Value = Id },
                        new SqlParameter("@TripID", SqlDbType.SmallInt) { Value = TripID }
                    };
                dsTripData = await _dbContext.get_Procedure_output_ds("SpGetTripRoutes", Login_params);

                return dsTripData;
            }
            catch (Exception ex)
            {

                throw new Exception(ex.Message);

            }
        }

        public async Task<OutData> GetConstructionSites()
        {
            try
            {
                OutData sites = new OutData();
                DbContextClass _dbContext = new DbContextClass(_config);
        
                SqlParameter[] Rparams =
                {
                        new SqlParameter("@ID", SqlDbType.VarChar) { Value = "" }
                    };
                sites = await _dbContext.get_Procedure_output_ds("SPGetConstructionSites", Rparams);
                return sites;
            }
            catch (Exception ex)
            {
                throw new Exception(ex.Message);
            }
        }

    }
}
