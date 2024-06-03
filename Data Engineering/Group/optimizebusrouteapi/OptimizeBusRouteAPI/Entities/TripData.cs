using System.ComponentModel.DataAnnotations;

namespace OptimizeBusRouteAPI.Entities
{
    public class TripData
    {
        [Key]
        public string route_id { get; set; }
        public string route_short_name { get; set; }
        public string route_long_name { get; set; }
        //public byte service_id { get; set; }
        //public Int16 trip_id { get; set; }
        //public string trip_headsign { get; set; }
        //public string? trip_short_name { get; set; }
        //public bool direction_id { get; set; }
        //public int block_id { get; set; }
        //public Int16 shape_id { get; set; }
        //public byte wheelchair_accessible { get; set; }
        //public byte bikes_allowed { get; set; }
    }
}
