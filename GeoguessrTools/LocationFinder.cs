using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using GeoguessrTools;

namespace GeoguessrTools
{
    public class CityLocation
    {
        public string CityName { get; set; }
        public double Latitude { get; set; }
        public double Longitude { get; set; }

        public CityLocation(string cityName, double latitude, double longitude)
        {
            CityName = cityName;
            Latitude = latitude;
            Longitude = longitude;
        }
    }

    public class LocationNameFinder
    {
        // List to hold the parsed city locations
        private List<CityLocation> _cities = new List<CityLocation>();

        /// <summary>
        /// Constructor: Loads city data from the JSON file.
        /// Make sure GlobalData.InputFilePath is set before calling this.
        /// </summary>
        public LocationNameFinder(string jsonPath)
        {
            // Load the JSON data using your provided method.
            JsonData root = GlobalData.ReadJson(jsonPath);

            // The root is expected to be an array. Loop over each JsonData element in the array.
            foreach (JsonData cityEntry in root.valueArr)
            {
                try
                {
                    // Extract values. It is assumed that the values are stored as strings.
                    string cityName = cityEntry.data["city"].value;
                    string latStr = cityEntry.data["lat"].value;
                    string lngStr = cityEntry.data["lng"].value;

                    // Parse latitude and longitude using InvariantCulture for consistency.
                    double latitude = double.Parse(latStr, CultureInfo.InvariantCulture);
                    double longitude = double.Parse(lngStr, CultureInfo.InvariantCulture);

                    _cities.Add(new CityLocation(cityName, latitude, longitude));
                }
                catch (Exception ex)
                {
                    // Optionally log or handle parsing errors for specific entries.
                    Console.WriteLine("Error parsing a city entry: " + ex.Message);
                }
            }
        }

        /// <summary>
        /// Finds the nearest city to the provided latitude and longitude.
        /// </summary>
        /// <param name="latitude">Latitude coordinate</param>
        /// <param name="longitude">Longitude coordinate</param>
        /// <returns>A tuple containing the city name and the distance (in kilometers) to that city.</returns>
        public string GetNearestCity(double latitude, double longitude)
        {
            CityLocation nearestCity = null;
            double minDistance = double.MaxValue;

            foreach (CityLocation city in _cities)
            {
                double distance = ComputeHaversineDistance(latitude, longitude, city.Latitude, city.Longitude);
                if (distance < minDistance)
                {
                    minDistance = distance;
                    nearestCity = city;
                }
            }

            // Return a tuple. If no city was found, the tuple will contain a null string and 0 distance.
            return minDistance < 5d ? nearestCity?.CityName : ("Umgebung " + nearestCity?.CityName);
        }

        /// <summary>
        /// Calculates the great-circle distance between two points using the Haversine formula.
        /// </summary>
        /// <param name="lat1">Latitude of the first point in decimal degrees.</param>
        /// <param name="lon1">Longitude of the first point in decimal degrees.</param>
        /// <param name="lat2">Latitude of the second point in decimal degrees.</param>
        /// <param name="lon2">Longitude of the second point in decimal degrees.</param>
        /// <returns>The distance in kilometers.</returns>
        private double ComputeHaversineDistance(double lat1, double lon1, double lat2, double lon2)
        {
            const double EarthRadiusKm = 6371; // Earth's radius in kilometers

            double dLat = ToRadians(lat2 - lat1);
            double dLon = ToRadians(lon2 - lon1);

            lat1 = ToRadians(lat1);
            lat2 = ToRadians(lat2);

            double a = Math.Pow(Math.Sin(dLat / 2), 2) +
                       Math.Cos(lat1) * Math.Cos(lat2) * Math.Pow(Math.Sin(dLon / 2), 2);

            double c = 2 * Math.Asin(Math.Sqrt(a));
            return EarthRadiusKm * c;
        }

        /// <summary>
        /// Converts degrees to radians.
        /// </summary>
        private double ToRadians(double degrees)
        {
            return degrees * (Math.PI / 180);
        }
    }
}
