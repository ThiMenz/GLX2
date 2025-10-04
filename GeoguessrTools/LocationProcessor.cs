using System;
using System.Collections.Generic;
using System.Formats.Asn1;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.Marshalling;

namespace GeoguessrTools
{
    /// <summary>
    /// Represents a tag occurrence with its computed percentage.
    /// </summary>
    public class TagInfo
    {
        public string Value { get; set; }
        public string Type { get; set; }
        public int Count { get; set; }
        public float Percentage { get; set; }
    }

    /// <summary>
    /// Holds detailed tag information and the top analyzed tags for a cluster.
    /// </summary>
    public class ClusterTagAnalysis
    {
        // All tags grouped by type and sorted by descending percentage
        public Dictionary<string, List<TagInfo>> TagTypeInfos { get; set; } = new();
        // Only the top tag(s) per type, based on the ratio threshold
        public Dictionary<string, List<TagInfo>> TopTagValues { get; set; } = new();
    }

    /// <summary>
    /// A node in the decision tree splitting clusters by TagType→TagValue.
    /// </summary>
    public class DecisionTreeNode
    {
        /// <summary>Which tag‐type is used to split at this node (null at root).</summary>
        public string TagType { get; set; }

        /// <summary>The tag‐value selected for this branch.</summary>
        public string TagValue { get; set; }

        /// <summary>Child nodes, one per distinct TagValue.</summary>
        public List<DecisionTreeNode> Children { get; } = new List<DecisionTreeNode>();

        /// <summary>If this is a leaf, the clusters that reach it.</summary>
        public List<string> Clusters { get; set; } = new List<string>();
    }

    public class JsonDataLocationProcessor
    {
        // Dictionary mapping cluster tag (e.g. "(25) Uummannaq") to a list of corresponding coordinate JsonData nodes.
        private Dictionary<string, List<JsonData>> _clusters = new Dictionary<string, List<JsonData>>();

        // Holds the complete JsonData of the input file.
        private JsonData _root;

        /// <summary>
        /// Constructor – reads the JSON file and groups each coordinate by its cluster tag.
        /// </summary>
        public JsonDataLocationProcessor()
        {
            // Read the input JSON as a JsonData tree.
            _root = GlobalData.ReadJson();

            // Expect that the input JSON object has a property "customCoordinates" holding an array.
            if (!_root.data.ContainsKey("customCoordinates"))
            {
                Console.WriteLine("No customCoordinates property found in the input JSON.");
                return;
            }

            // Iterate over the coordinates (each is a JsonData node within the array).
            foreach (JsonData coordinate in _root.data["customCoordinates"].valueArr)
            {
                // Look for the cluster tag by checking the extra->tags array.
                if (coordinate.data.ContainsKey("extra") &&
                    coordinate.data["extra"].data.ContainsKey("tags"))
                {
                    JsonData tagsNode = coordinate.data["extra"].data["tags"];
                    string clusterTag = null;

                    // tags are stored as an array in our JsonData representation.
                    foreach (JsonData tagData in tagsNode.valueArr)
                    {
                        if (tagData.value.StartsWith("("))
                        {
                            clusterTag = tagData.value;
                            break;
                        }
                    }

                    if (clusterTag != null)
                    {
                        if (!_clusters.ContainsKey(clusterTag))
                        {
                            _clusters[clusterTag] = new List<JsonData>();
                        }
                        _clusters[clusterTag].Add(coordinate);
                    }
                }
            }
        }

        public void AnalyzeClusterTags()
        {
            // Final structured result for all clusters
            var allAnalysis = new Dictionary<string, ClusterTagAnalysis>();

            // Iterate each cluster
            foreach (var kvp in _clusters)
            {
                string clusterName = kvp.Key;
                List<JsonData> cluster = kvp.Value;
                var analysis = new ClusterTagAnalysis();

                // Temporary storage: tagType -> (tagValue -> count)
                var counts = new Dictionary<string, Dictionary<string, int>>();

                // Collect raw counts
                foreach (var loc in cluster)
                {
                    foreach (var jtag in loc.G("extra").G("tags").valueArr)
                    {
                        string tagString = jtag.value;
                        if (!tagString.StartsWith("§"))
                            continue;

                        //Console.WriteLine(tagString);

                        // Parse out type and value
                        int braceOpen = tagString.LastIndexOf('{');
                        int braceClose = tagString.LastIndexOf('}');
                        if (braceOpen < 0 || braceClose < 0 || braceClose < braceOpen)
                            continue;



                        string tagType = tagString.Substring(braceOpen + 1, braceClose - braceOpen - 1).Trim();
                        string tagValue = tagString.Substring(0, braceOpen).Trim();

                        if (!counts.ContainsKey(tagType))
                            counts[tagType] = new Dictionary<string, int>();

                        if (!counts[tagType].ContainsKey(tagValue))
                            counts[tagType][tagValue] = 0;

                        counts[tagType][tagValue]++;
                    }
                }

                int totalLocations = cluster.Count;

                // Compute percentages, sort, and apply analysis per type
                foreach (var typeEntry in counts)
                {
                    string tagType = typeEntry.Key;
                    // Build list of TagInfo with percentage
                    var tagList = typeEntry.Value
                        .Select(tv => new TagInfo
                        {
                            Type = tagType,
                            Value = tv.Key,
                            Count = tv.Value,
                            Percentage = (float)tv.Value / totalLocations
                        })
                        .OrderByDescending(ti => ti.Percentage)
                        .ToList();

                    analysis.TagTypeInfos[tagType] = tagList;

                    // Select top tag(s) based on ratio threshold
                    var topTags = new List<TagInfo>();
                    if (tagList.Count > 0)
                    {
                        TagInfo top = tagList[0];
                        topTags.Add(top);

                        foreach (var candidate in tagList.Skip(1))
                        {
                            if (top.Percentage / candidate.Percentage < 1.33f) //!!!
                                topTags.Add(candidate);
                            else
                                break; // further ones will be even smaller
                        }
                    }

                    analysis.TopTagValues[tagType] = topTags;
                }

                allAnalysis[clusterName] = analysis;
            }

            foreach (var clusterEntry in allAnalysis)
            {
                var analysis = clusterEntry.Value;

                foreach (var typeEntry in analysis.TopTagValues)
                {
                    //Console.WriteLine(typeEntry.Key);
                    foreach (var ti in typeEntry.Value)
                    {

                    }
                }
            }

            DecisionTreeNode dtn = BuildDecisionTree(allAnalysis, new List<string> { "cs", "mls" });

            PrintDecisionTree(dtn);

            // Nicely print the analyzed results
            /*foreach (var clusterEntry in allAnalysis)
            {
                Console.WriteLine($"Cluster: {clusterEntry.Key}");
                var analysis = clusterEntry.Value;

                foreach (var typeEntry in analysis.TopTagValues)
                {
                    Console.WriteLine($"  Tag Type: {typeEntry.Key}");
                    foreach (var ti in typeEntry.Value)
                    {
                        Console.WriteLine($"    {ti.Value}: {ti.Percentage:P1} ({ti.Count}/{_clusters[clusterEntry.Key].Count})");
                    }
                }

                Console.WriteLine();
            }*/
        }

        /// <summary>
        /// Build a decision tree over the clusters in allAnalysis.
        /// At depth d, splits on splitTagTypes[d].  
        /// </summary>
        public DecisionTreeNode BuildDecisionTree(
            Dictionary<string, ClusterTagAnalysis> allAnalysis,
            IList<string> splitTagTypes)
        {
            var root = new DecisionTreeNode();
            // start recursion with _all_ cluster names
            BuildNode(root,
                      allAnalysis.Keys.ToList(),
                      allAnalysis,
                      splitTagTypes,
                      0);
            return root;
        }

        private void BuildNode(
    DecisionTreeNode node,
    List<string> clusterNames,
    Dictionary<string, ClusterTagAnalysis> allAnalysis,
    IList<string> splitTagTypes,
    int depth)
        {
            // stopping criteria
            if (depth >= splitTagTypes.Count || clusterNames.Count <= 1)
            {
                node.Clusters = clusterNames;
                return;
            }

            string tagType = splitTagTypes[depth];
            var groups = new Dictionary<string, List<string>>();

            foreach (var cname in clusterNames)
            {
                var analysis = allAnalysis[cname];
                List<string> tagValues;

                // gather _all_ top values for this tagType
                if (analysis.TopTagValues.TryGetValue(tagType, out var infos) && infos.Count > 0)
                {
                    tagValues = infos.Select(ti => ti.Value).ToList();
                }
                else
                {
                    tagValues = new List<string> { "<none>" };
                }

                // assign this cluster into each applicable group
                foreach (var val in tagValues)
                {
                    if (val.ToString().Contains("?")) continue;
                    if (!groups.ContainsKey(val))
                        groups[val] = new List<string>();
                    groups[val].Add(cname);
                }
            }

            // now build one child per distinct value
            foreach (var kvp in groups)
            {
                var child = new DecisionTreeNode
                {
                    TagType = tagType,
                    TagValue = kvp.Key
                };
                node.Children.Add(child);
                BuildNode(child, kvp.Value, allAnalysis, splitTagTypes, depth + 1);
            }
        }

        /// <summary>
        /// Nicely print the tree to the console with indentation.
        /// </summary>
        public void PrintDecisionTree(DecisionTreeNode node, int indent = 0)
        {
            string pad = new string(' ', indent * 2);
            if (node.TagType != null)
            {
                Console.WriteLine($"{pad}{node.TagType} = {node.TagValue}");
            }

            if (node.Children.Count > 0)
            {
                foreach (var child in node.Children)
                    PrintDecisionTree(child, indent + 1);
            }
            else
            {
                // leaf: show which clusters ended up here
                Console.WriteLine($"{pad}  Clusters: {string.Join(", ", node.Clusters)}");
            }
        }

        public void Test()
        {
            foreach (string clusterName in _clusters.Keys)
            {
                Console.WriteLine(clusterName);
                List<JsonData> cluster = _clusters[clusterName];
                Dictionary<string, int> absoluteTagCounts = new();
                foreach (JsonData loc in cluster)
                {
                    foreach (JsonData jtag in loc.G("extra").G("tags").valueArr)
                    {
                        string tag = jtag.value;
                        if (!tag.StartsWith("§")) continue;
                        if (!absoluteTagCounts.ContainsKey(tag)) absoluteTagCounts.Add(tag, 1);
                        else absoluteTagCounts[tag] += 1;
                    }
                }

                foreach (string tag in absoluteTagCounts.Keys)
                {
                    int count = absoluteTagCounts[tag];
                    Console.WriteLine(tag + ": " + ((float)count / cluster.Count).ToString());
                }

                Console.WriteLine();
            }
        }

        /// <summary>
        /// Processes the coordinate nodes by randomly selecting up to n locations from each cluster.
        /// Builds a new JsonData object matching the input format.
        /// </summary>
        /// <param name="n">Number of locations to select per cluster.</param>
        /// <returns>A new JsonData instance with the filtered locations.</returns>
        public JsonData ProcessRandomSelection(int n)
        {
            // Create a new JsonData node for the "customCoordinates" array.
            JsonData newCustomCoordinates = new JsonData(null)
            {
                holdsArray = true
            };

            Random rnd = new Random();

            // For each cluster group, randomize and take up to n locations.
            foreach (var cluster in _clusters)
            {
                List<JsonData> group = cluster.Value;
                // Shuffle the group using OrderBy with a random key.
                List<JsonData> shuffled = group.OrderBy(x => rnd.Next()).ToList();

                // Take up to n coordinates from this cluster.
                foreach (JsonData selected in shuffled.Take(n))
                {
                    newCustomCoordinates.AppendValueArray(selected);
                }
            }

            // Create the output JsonData structure. We follow the same structure as the input.
            JsonData output = new JsonData(null);

            // Copy over the "name" property if it exists; otherwise, use a default.
            if (_root.data.ContainsKey("name"))
            {
                output.AppendPropertyDict("name", _root.data["name"]);
            }
            else
            {
                output.AppendPropertyDict("name", new JsonData(null).SetValue("Filtered Locations"));
            }

            // Assign our newly created array to "customCoordinates".
            output.AppendPropertyDict("customCoordinates", newCustomCoordinates);

            return output;
        }

        /// <summary>
        /// Saves the given JsonData output to the file path specified by GlobalData.OutputFilePath.
        /// </summary>
        /// <param name="output">The output JsonData object.</param>
        public void SaveToFile(JsonData output)
        {
            // Use the JsonData.ToString() to convert the structure to a JSON-formatted string.
            string jsonOutput = output.ToString();
            File.WriteAllText(GlobalData.OutputFilePath, jsonOutput);
            Console.WriteLine($"File successfully written to: {GlobalData.OutputFilePath}");
        }
    }

}