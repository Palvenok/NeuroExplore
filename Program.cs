using System.Drawing;
using System.IO.Compression;

class Program
{
    static void Main()
    {
        int imageWidth = 90;
        int imageHeight = 140;
        int pixelsPerImage = imageWidth * imageHeight;

        Dictionary<int, double[][]> inputs = new Dictionary<int, double[][]>();
        
        var nn = new NeuralNetwork();
        if(File.Exists("model.zip")) 
        {
            nn = NetworkStorage.LoadModel("model.zip");
            Console.WriteLine("Model was loaded!");
        }
        else
        {
            nn.AddLayer(12600, 32); // скрытый слой
            nn.AddLayer(32, 16); // скрытый слой
            nn.AddLayer(16, 10); // скрытый слой
            nn.AddLayer(10, 1, ActivationType.Linear); // выходной слой с линейной активацией
            Console.WriteLine("Created new model!");
        }

        double[] input = ProcessImage(@"image.jpg");

        for (int data = 0; data <= 9; data++)
        {
            string binPath = $@"TrainData/{data}.bin"; // путь к файлу

            using (BinaryReader reader = new BinaryReader(File.Open(binPath, FileMode.Open)))
            {
                long totalFloats = reader.BaseStream.Length / sizeof(float);
                int imageCount = (int)(totalFloats / pixelsPerImage);

                inputs.Add(data, new double[imageCount][]);

                for (int i = 0; i < imageCount; i++)
                {
                    inputs[data][i] = new double[pixelsPerImage];
                    for (int j = 0; j < pixelsPerImage; j++)
                    {
                        inputs[data][i][j] = reader.ReadSingle();
                    }
                }
            }

            Console.WriteLine($"Прочитано {inputs[data].Length} изображений из {binPath}");
        }
            
        for (int epoch = 0; epoch < 10; epoch++)
        {
            Console.WriteLine($"Epoch {epoch}");
            double error = 0;
            for (int data = 0; data <= 9; data++)
            {
                for (int i = 0; i < inputs[data].Length; i++)
                    error += nn.Train(inputs[data][i], new double[] {data});
                Console.WriteLine($"\tData {data}, Error: {error:F4}");
            }
            if (epoch % 1 == 0)
                Console.WriteLine($"Epoch {epoch}, Error: {error:F4}");
            Console.WriteLine();
        }

        

        Console.WriteLine("Сохранение модели:");
        NetworkStorage.Save("model.zip", nn);

        var output = nn.Predict(input);
        Console.WriteLine("is: " + output[0]);
        Console.ReadKey();
    }
    
    static double[] ProcessImage(string imagePath)
    {
        using (Bitmap bmp = new Bitmap(imagePath))
        {
            double[] pixels = new double[bmp.Width * bmp.Height];
            int index = 0;

            for (int y = 0; y < bmp.Height; y++)
            {
                for (int x = 0; x < bmp.Width; x++)
                {
                    Color pixel = bmp.GetPixel(x, y);
                    float gray = (pixel.R + pixel.G + pixel.B) / 3f / 255f;
                    pixels[index++] = 1f - gray;
                }
            }

            return pixels;
        }
    }
}