using GeoguessrTools;

/*
GlobalData.InputFilePath = @"C:\Users\tpmen\Downloads\Copy & Paste.json";
GlobalData.OutputFilePath = @"C:\Users\tpmen\clustered-GLtest-locations-panos.json";
Clustering cluster = new Clustering();
cluster.SimpleClusteringProcess(0.02d, new LocationNameFinder(@"C:\Users\tpmen\Downloads\gl.json"));
*/

/*
GlobalData.InputFilePath = @"C:\Users\tpmen\clustered-GLtest-locations.json";
GlobalData.OutputFilePath = @"C:\Users\tpmen\balanced-GLtest-locations.json";
JsonDataLocationProcessor locationProcessor = new JsonDataLocationProcessor();
JsonData data = locationProcessor.ProcessRandomSelection(40);
locationProcessor.SaveToFile(data);
*/


GlobalData.InputFilePath = @"C:\Users\tpmen\Downloads\SN_GLX2v2.json";
GlobalData.OutputFilePath = @"C:\Users\tpmen\Downloads\SN_GLX2v2AUTOCLUSTERS.json";

Clustering cluster = new Clustering();
cluster.RoadClusteringProcess(new List<string> { "paved", "clouds", "roadlines", "carcolor", "antenna" });

//GlobalData.OutputFilePath = @"C:\Users\tpmen\balanced-GLtest-locations.json";
//JsonDataLocationProcessor locationProcessor = new JsonDataLocationProcessor();
//locationProcessor.RoadClusteringProcess();
//locationProcessor.SaveToFile(data);
