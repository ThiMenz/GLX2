using System.ComponentModel.DataAnnotations;
using System.Drawing;
using System.Globalization;
using System.Linq;

namespace GeoguessrTools
{
    public class GraphNode
    {
        public JsonData nodeData;
        public double lat, lng;
        public Dictionary<string, string> tags = new();
        public List<GraphNode> neighboringNodes = new();

        public static List<GraphNode> fullGraph = new();

        public GraphNode(JsonData pNodeData)
        {
            nodeData = pNodeData;
            fullGraph.Add(this);
            lat = ParseDouble(nodeData.G("lat").value);
            lng = ParseDouble(nodeData.G("lng").value);

            JsonData ttags = nodeData.G("extra").G("tags");
            for (int j = 0; j < ttags.Len(); j++)
            {
                string tStr = ttags.G(j).value;
                if (ttags.G(j).value.StartsWith("§"))
                {
                    string[] tagSpl = tStr.Split("=");
                    if (tagSpl.Length == 2)
                    tags.Add(tagSpl[0].Replace("§", ""), tagSpl[1]);
                }
            }
        }

        public static void InitializeEntireGraph(double maxConnectionDistance = 0.0035)
        {
            foreach (GraphNode node in fullGraph)
            {
                foreach (GraphNode potentialNeighbor in fullGraph)
                {
                    if (node == potentialNeighbor) continue;

                    if (node.DistanceTo(potentialNeighbor) > maxConnectionDistance) continue;

                    node.neighboringNodes.Add(potentialNeighbor);
                }
            }
        }

        public double DistanceTo(GraphNode g2)
        {
            return (g2.lng - lng) * (g2.lng - lng) + (g2.lat - lat) * (g2.lat - lat);
        }

        private double ParseDouble(string str)
        {
            return double.Parse(str, NumberStyles.Number, CultureInfo.InvariantCulture);
        }
    }

    public class Clustering
    {
        public void RoadClusteringProcess(List<string> relevantTags)
        {
            JsonData data = GlobalData.ReadJson();
            JsonData allCoordinates = data.G("customCoordinates");

            for (int i = 0; i < allCoordinates.Len(); i++) _ = new GraphNode(data.G("customCoordinates").G(i));
            GraphNode.InitializeEntireGraph();

            Dictionary<GraphNode, bool> unpriorizedForRandomSelection = new Dictionary<GraphNode, bool>();
            Dictionary<GraphNode, bool> classified = new Dictionary<GraphNode, bool>();

            int iteration = 0;

            while (true)
            {
                iteration++;
                GraphNode? selectedStartingPoint = null;
                foreach (GraphNode node in GraphNode.fullGraph)
                {
                    if (unpriorizedForRandomSelection.ContainsKey(node) || classified.ContainsKey(node)) continue;

                    selectedStartingPoint = node;
                }

                if (selectedStartingPoint == null)
                {
                    foreach (GraphNode node in unpriorizedForRandomSelection.Keys)
                    {
                        selectedStartingPoint = node;
                        unpriorizedForRandomSelection.Remove(node);
                        break;
                    }
                }

                if (selectedStartingPoint == null) break;

                classified[selectedStartingPoint] = true;

                Dictionary<GraphNode, bool> tmpFloodfillBlocked = new Dictionary<GraphNode, bool>(), finalCluster = new();
                finalCluster.Add(selectedStartingPoint, true);
                List<GraphNode> floodfillFrontiers = new() { selectedStartingPoint };

                while (floodfillFrontiers.Count != 0)
                {
                    List<GraphNode> newFrontiers = new();
                    foreach (GraphNode node in floodfillFrontiers)
                    {
                        foreach (GraphNode neighbor in node.neighboringNodes)
                        {
                            if (finalCluster.ContainsKey(neighbor) || classified.ContainsKey(neighbor)) continue;

                            classified.Add(neighbor, true);
                            finalCluster.Add(neighbor, true);

                            bool sameProps = true;
                            foreach (string tag in relevantTags) if (!neighbor.tags.ContainsKey(tag) || neighbor.tags[tag] != selectedStartingPoint.tags[tag]) sameProps = false;
                            if (!sameProps) continue;

                            newFrontiers.Add(neighbor);
                        }
                    }
                    floodfillFrontiers = newFrontiers.ToArray().ToList();
                    newFrontiers.Clear();
                }
                string clusterName = "[Cluster " + iteration + "]";

                foreach (GraphNode gn in finalCluster.Keys)
                {
                    gn.nodeData.G("extra").G("tags").AppendValueArray(new JsonData(null).SetValue("[Cluster " + iteration + "]"));
                } 
            }

            Console.WriteLine(String.Format("Found {0} Clusters!", iteration - 1));

            File.WriteAllText(GlobalData.OutputFilePath, data.ToString());
        }

        public void SimpleClusteringProcess(double clusterSize, LocationNameFinder lmf) //SQUARED
        {
            JsonData data = GlobalData.ReadJson();
            JsonData allCoordinates = data.G("customCoordinates");

            List<List<(double, double)>> clusters = new List<List<(double, double)>>();

            bool allClustersFinalized = false;
            Dictionary<(int, int), bool> tagMerges = new Dictionary<(int, int), bool>();

            for (int i = 0; i < allCoordinates.Len(); i++)
            {
                JsonData loc = data.G("customCoordinates").G(i);
                double lat = ParseDouble(loc.G("lat").value), lng = ParseDouble(loc.G("lng").value);

                bool b = true;
                int a = 0;
                List<int> allClusters = new List<int>();
                foreach (List<(double, double)> list in clusters)
                {
                    foreach ((double, double) pos in list)
                    {
                        double v1 = lat - pos.Item1, v2 = lng - pos.Item2;
                        if (Math.Sqrt(v1 * v1 + v2 * v2) < clusterSize)
                        {
                            loc.G("extra").G("tags").AppendValueArray(new JsonData(null).SetValue("_T" + a.ToString()));
                            b = false;
                            clusters[a].Add((lat, lng));
                            allClusters.Add(a);
                            break;
                        }
                    }
                    a++;
                }

                if (b)
                {
                    loc.G("extra").G("tags").AppendValueArray(new JsonData(null).SetValue("_T" + clusters.Count.ToString()));
                    clusters.Add(new List<(double, double)>() { (lat, lng) });
                }
            }

            while (!allClustersFinalized)
            {
                allClustersFinalized = true;

                Dictionary<(string, string), bool> tagReplacements = new();
                for (int i = 0; i < allCoordinates.Len(); i++)
                {
                    JsonData tags = data.G("customCoordinates").G(i).G("extra").G("tags");
                    string firstTag = "";
                    for (int j = 0; j < tags.Len(); j++)
                    {
                        string tStr = tags.G(j).value;
                        if (tags.G(j).value.StartsWith("_"))
                        {
                            if (firstTag == "") firstTag = tStr;
                            else tagReplacements.TryAdd((firstTag, tStr), false);
                        }
                    }
                }

                List<(string, string)> tagMergeArrSorted = tagReplacements.Keys.ToList();
                tagMergeArrSorted.Sort((a, b) => int.Parse(b.Item1.Replace("_T", "")).CompareTo(int.Parse(a.Item1.Replace("_T", ""))));

                foreach ((string, string) val in tagMergeArrSorted)
                {
                    //Console.WriteLine(val);
                    //string townTagNew = "_T" + val.Item1, townTagOld = "_T" + val.Item2;
                    for (int i = 0; i < allCoordinates.Len(); i++)
                    {
                        ChangeTownTag(data.G("customCoordinates").G(i), val.Item2, val.Item1);
                    }
                }

                for (int i = 0; i < allCoordinates.Len(); i++)
                {
                    JsonData tags = data.G("customCoordinates").G(i).G("extra").G("tags");
                    string firstTag = "";
                    for (int j = 0; j < tags.Len(); j++)
                    {
                        string tStr = tags.G(j).value;
                        if (tags.G(j).value.StartsWith("_"))
                        {
                            if (firstTag == "") firstTag = tStr;
                            else allClustersFinalized = false;
                        }
                    }

                    if (!allClustersFinalized) break;
                }

                Console.WriteLine("ITERATION FINISHED");
            }

            Dictionary<string, bool> finalizedClusterTags = new Dictionary<string, bool>();
            List<int> exampleLocIDs = new List<int>();
            for (int i = 0; i < allCoordinates.Len(); i++)
            {
                JsonData tags = data.G("customCoordinates").G(i).G("extra").G("tags");
                for (int j = 0; j < tags.Len(); j++)
                    if (tags.G(j).value.StartsWith("_"))
                        if (finalizedClusterTags.TryAdd(tags.G(j).value, false))
                            exampleLocIDs.Add(i);

                if (!allClustersFinalized) break;
            }

            string[] clusterTagsFin = finalizedClusterTags.Keys.ToArray();
            int size = clusterTagsFin.Length.ToString().Length;
            for (int i = 0; i < finalizedClusterTags.Count; i++)
            {
                JsonData exampleLoc = allCoordinates.G(exampleLocIDs[i]);
                string curName = "(" + GetUIntStringWithZeros(i, size) + ") " + 
                    lmf.GetNearestCity(ParseDouble(exampleLoc.G("lat").value), ParseDouble(exampleLoc.G("lng").value)), clusterTag = clusterTagsFin[i];
                for (int j = 0; j < allCoordinates.Len(); j++)
                {
                    ChangeTownTag(data.G("customCoordinates").G(j), clusterTag, curName);
                }
            }

            File.WriteAllText(GlobalData.OutputFilePath, data.ToString());
        }

        private string GetUIntStringWithZeros(int val, int size)
        {
            string tStr = val.ToString();
            for (int i = tStr.Length; i < size; i++) tStr = "0" + tStr;
            return tStr;
        }

        private void ChangeTownTag(JsonData loc, string oldTag, string newTag)
        {
            JsonData tags = loc.G("extra").G("tags");
            if (tags.Contains(oldTag))
            {
                tags.Remove(oldTag);
                if (!tags.Contains(newTag)) tags.AppendValueArray(new JsonData(null).SetValue(newTag));
            }
        }

        private double ParseDouble(string str)
        {
            return double.Parse(str, NumberStyles.Number, CultureInfo.InvariantCulture);
        }
    }
}
