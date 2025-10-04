using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;

namespace GeoguessrTools
{

    public static class GlobalData
    {
        public static string InputFilePath = "";
        public static string OutputFilePath = "";

        public static JsonData ReadJson(string path)
        {
            string jsonContent = File.ReadAllText(path);
            using JsonDocument document = JsonDocument.Parse(jsonContent);
            JsonElement root = document.RootElement;
            JsonData result = ParseElement(root, null);

            return result;
        }

        public static JsonData ReadJson()
        {
            string jsonContent = File.ReadAllText(InputFilePath);
            using JsonDocument document = JsonDocument.Parse(jsonContent);
            JsonElement root = document.RootElement;
            JsonData result = ParseElement(root, null);

            return result;
        }

        private static JsonData ParseElement(JsonElement element, JsonData parent)
        {
            JsonData node = new JsonData(parent);

            switch (element.ValueKind)
            {
                case JsonValueKind.Object:
                    foreach (JsonProperty property in element.EnumerateObject())
                    {
                        // Rekursiver Aufruf für jedes Property
                        node.AppendPropertyDict(property.Name, ParseElement(property.Value, node));
                    }
                    break;

                case JsonValueKind.Array:
                    node.holdsArray = true;
                    foreach (JsonElement item in element.EnumerateArray())
                    {
                        // Rekursiver Aufruf für jedes Array-Element
                        node.AppendValueArray(ParseElement(item, node));
                    }
                    break;

                // Bei Leaf-Werten wird der Wert als String gespeichert.
                case JsonValueKind.String:
                    node.SetValue(element.GetString());
                    break;

                case JsonValueKind.Number:
                    // Hier wird GetRawText() verwendet, um die Zahl exakt als im JSON zu erhalten
                    node.SetValue(element.GetRawText());
                    break;

                case JsonValueKind.True:
                case JsonValueKind.False:
                    node.SetValue(element.GetBoolean().ToString());
                    break;

                case JsonValueKind.Null:
                    node.SetValue("null");
                    break;
            }

            return node;
        }
    }

    public class JsonData
    {
        public bool holdsArray = false;
        public bool isLeafValue = false;
        public string value = "";
        public List<JsonData> valueArr = new List<JsonData>();
        public Dictionary<string, JsonData> data = new();
        public JsonData parent;

        public JsonData(JsonData parent)
        {
            this.parent = parent;
        }

        public bool Contains(string el)
        {
            foreach (JsonData jd in valueArr)
                if (jd.value == el) return true;
            return false;
        }

        public void Remove(string el)
        {
            for (int i = 0; i < valueArr.Count; i++)
                if (valueArr[i].value == el) valueArr.RemoveAt(i--);
        }

        public int Len()
        {
            return valueArr.Count;
        }

        public JsonData G(string tVal)
        {
            if (holdsArray) throw new Exception();
            return data[tVal];
        }

        public JsonData G(int idx)
        {
            if (!holdsArray) throw new Exception();
            return valueArr[idx];
        }

        public JsonData SetValue(string val)
        {
            value = val;
            isLeafValue = true;
            return this;
        }

        public void AppendPropertyDict(string key, JsonData val)
        {
            data.Add(key, val);
            val.parent = this;
        }

        public void AppendValueArray(JsonData val, bool sort = false)
        {
            if (isLeafValue)
                throw new Exception("Cannot add array elements to a leaf value!");
            holdsArray = true;
            valueArr.Add(val);
            val.parent = this;
            if (sort) valueArr.Sort((a, b) => a.value.StartsWith("_T") && b.value.StartsWith("_T") ? int.Parse(b.value.Replace("_T", "")).CompareTo(int.Parse(a.value.Replace("_T", ""))) : 0);
        }

        private bool IsNumber(string str)
        {
            int a = 0;
            foreach (char ch in str)
            {
                a++;
                if (Char.IsNumber(ch) || ch == '.' || ch == ',' || (a == 1 && ch == '-')) continue;
                else return false;
            }
            return true;
        }

        public override string ToString()
        {
            // Wenn es ein einfacher (Leaf-) Wert ist, wird dieser zurückgegeben.
            if (isLeafValue)
            {
                if (IsNumber(value) || value == "null") return value;
                else return '"' + value + '"';
            }
            // Bei Arrays bauen wir eine eckige Klammer-Notation.
            else if (holdsArray)
            {
                StringBuilder sb = new StringBuilder();
                sb.Append("[");
                for (int i = 0; i < valueArr.Count; i++)
                {
                    sb.Append(valueArr[i].ToString());
                    if (i < valueArr.Count - 1)
                        sb.Append(", ");
                }
                sb.Append("]");
                return sb.ToString();
            }
            // Bei Objekten werden alle Properties rekursiv ausgegeben.
            else if (data.Count > 0)
            {
                StringBuilder sb = new StringBuilder();
                sb.Append("{");
                int i = 0;
                foreach (var kvp in data)
                {
                    sb.Append($"\"{kvp.Key}\": {kvp.Value.ToString()}");
                    if (i < data.Count - 1)
                        sb.Append(", ");
                    i++;
                }
                sb.Append("}");
                return sb.ToString();
            }
            else
            {
                return "{}";
            }
        }
    }

}
