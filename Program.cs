class Program
{
    static void Main()
    {
        
        double[][] inputs = {
            new double[] { 0.1, 0.2 },
            new double[] { 0.5, 0.7 },
            new double[] { 0.9, 0.05 },
            new double[] { 0.3, 0.3 },
            new double[] { 0.39, 0.84 },
            new double[] { 1.7, 0.33 },
            new double[] { 0.01, 0.5 },
            new double[] { 0.02, 0.03 },
            new double[] { 0.72, 0.86 }
        };

        double[][] outputs = {
            new double[] { 0.3 },
            new double[] { 1.2 },
            new double[] { 0.95 },
            new double[] { 0.6 },
            new double[] { 1.23 },
            new double[] { 2.03 },
            new double[] { 0.51 },
            new double[] { 0.05 },
            new double[] { 1.58 }
        };
        
        var nn = new NeuralNetwork();
        if(File.Exists("model.zip")) 
            nn = NetworkStorage.LoadModel("model.zip");
        else
        {
            nn.AddLayer(2, 6); // скрытый слой
            nn.AddLayer(6, 4); // скрытый слой
            nn.AddLayer(4, 1, ActivationType.Linear); // выходной слой с линейной активацией
        }

        for (int epoch = 0; epoch < 10000; epoch++)
        {
            double error = 0;
            for (int i = 0; i < inputs.Length; i++)
                error += nn.Train(inputs[i], outputs[i]);

            if (epoch % 1000 == 0)
                Console.WriteLine($"Epoch {epoch}, Error: {error:F4}");
        }

        Console.WriteLine("\nРезультаты:");
        foreach (var input in inputs)
        {
            var prediction = nn.Predict(input);
            Console.WriteLine($"{input[0]} + {input[1]} ≈ {prediction[0]:F4}");
        }
        
        var p = nn.Predict(new double[] { 0.98, 1.02 });
        Console.WriteLine($"0.98 + 1.02 ≈ {p[0]:F4}");

        NetworkStorage.Save("model.zip", nn);
        Console.ReadKey();
    }
}
