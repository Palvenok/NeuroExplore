using System;
using System.Globalization;
using System.IO;
using System.IO.Compression;

public static class NetworkStorage
{
    public static void Save(string zipPath, NeuralNetwork network)
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "nn_layers_" + Guid.NewGuid());
        Directory.CreateDirectory(tempDir);

        for (int i = 0; i < network.Layers.Count; i++)
        {
            string csvPath = Path.Combine(tempDir, $"layer_{i}.csv");
            using (var writer = new StreamWriter(csvPath))
            {
                var layer = network.Layers[i];
                for (int row = 0; row < layer.OutputSize; row++)
                {
                    for (int col = 0; col < layer.InputSize; col++)
                    {
                        writer.Write(layer.Weights[row, col].ToString(CultureInfo.InvariantCulture));
                        if (col < layer.InputSize - 1)
                            writer.Write(",");
                    }
                    writer.WriteLine();
                }

                // Сохраняем смещения как последнюю строку
                for (int iBias = 0; iBias < layer.Biases.Length; iBias++)
                {
                    writer.Write(layer.Biases[iBias].ToString(CultureInfo.InvariantCulture));
                    if (iBias < layer.Biases.Length - 1)
                        writer.Write(",");
                }
                writer.WriteLine();
            }
        }

        // Сохраняем структуру сети
        string metaPath = Path.Combine(tempDir, "meta.txt");
        using (var writer = new StreamWriter(metaPath))
        {
            writer.WriteLine(network.Layers.Count);
            foreach (var layer in network.Layers)
                writer.WriteLine($"{layer.InputSize} {layer.OutputSize} {(int)layer.ActivationKind}");
        }

        if (File.Exists(zipPath))
            File.Delete(zipPath);

        ZipFile.CreateFromDirectory(tempDir, zipPath);
        Directory.Delete(tempDir, true);
    }

    public static NeuralNetwork LoadModel(string zipPath)
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "nn_layers_" + Guid.NewGuid());
        ZipFile.ExtractToDirectory(zipPath, tempDir);

        string metaPath = Path.Combine(tempDir, "meta.txt");
        var metaLines = File.ReadAllLines(metaPath);
        int layerCount = int.Parse(metaLines[0]);

        var network = new NeuralNetwork();
        for (int i = 0; i < layerCount; i++)
        {
            var parts = metaLines[i + 1].Split();
            int inputSize = int.Parse(parts[0]);
            int outputSize = int.Parse(parts[1]);
            ActivationType activation = (ActivationType)Int32.Parse(parts[2]);
            network.AddLayer(inputSize, outputSize, activation);
        }

        for (int i = 0; i < network.Layers.Count; i++)
        {
            string csvPath = Path.Combine(tempDir, $"layer_{i}.csv");
            var lines = File.ReadAllLines(csvPath);
            var layer = network.Layers[i];

            for (int row = 0; row < layer.OutputSize; row++)
            {
                var values = lines[row].Split(',');
                for (int col = 0; col < values.Length; col++)
                {
                    layer.Weights[row, col] = double.Parse(values[col], CultureInfo.InvariantCulture);
                }
            }

            // Последняя строка — смещения
            var biasValues = lines[layer.OutputSize].Split(',');
            for (int b = 0; b < biasValues.Length; b++)
            {
                layer.Biases[b] = double.Parse(biasValues[b], CultureInfo.InvariantCulture);
            }
        }

        Directory.Delete(tempDir, true);
        return network;
    }
}
