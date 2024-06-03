using OptimizeBusRouteAPI.Entities;
using Microsoft.EntityFrameworkCore;
using Microsoft.Data.SqlClient;
using System.Data;

namespace OptimizeBusRouteAPI.Data
{
    public class DbContextClass : DbContext
    {
        protected readonly IConfiguration Configuration;

        public DbContextClass(IConfiguration configuration)
        {
            Configuration = configuration;
        }
        protected override void OnConfiguring(DbContextOptionsBuilder options)
        {
            options.UseSqlServer(Configuration.GetConnectionString("connectionString"));
        }

        public class OutList
        {
            public object? Keys { get; set; }
            public object? values { get; set; }

        }

        public class OutData
        {
            public List<OutList>? OutParamData { get; set; }
            public DataSet? ResponseData;
        }

        public DbSet<TripData> Trips { get; set; }


        public Task<OutData> get_Procedure_output_ds(string procnames, SqlParameter[] param)
        {
            try
            {
                string Con = Configuration["connectionString"];
                OutData datas;
                OutList lst = null;
                List<OutList> lsts = null;
                using (SqlConnection con = new SqlConnection(Con))
                {
                    con.Open();
                    using (SqlCommand cmd = new SqlCommand(procnames, con))
                    {
                        cmd.CommandType = CommandType.StoredProcedure;
                        cmd.CommandTimeout = 0;
                        cmd.Parameters.AddRange(param);
                        DataSet tables = new DataSet();
                        using (SqlDataAdapter dataAdapter = new SqlDataAdapter())
                        {

                            dataAdapter.SelectCommand = cmd;
                            dataAdapter.Fill(tables);
                            lsts = new List<OutList>();
                            for (int i = 0; i < cmd.Parameters.Count; i++)
                            {

                                lst = new OutList()
                                {
                                    values = cmd.Parameters[i].Value == null ? null : cmd.Parameters[i].Value.ToString(),
                                    Keys = cmd.Parameters[i].ParameterName
                                };
                                lsts.Add(lst);
                            }
                        }

                        datas = new OutData()
                        {
                            OutParamData = lsts,
                            ResponseData = tables
                        };
                    }
                }

                return Task.Run(() =>
                {
                    return datas;
                });
            }
            catch (Exception ex)
            {
                throw;
            }
        }
    }
}
