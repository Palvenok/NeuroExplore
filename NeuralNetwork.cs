public class NeuralNetwork
{
    private List<Layer> layers = new List<Layer>();
    private double learningRate = 0.1;

    public List<Layer> Layers => layers;

    public void AddLayer(int inputSize, int outputSize, ActivationType activationType = ActivationType.Sigmoid)
    {
        layers.Add(new Layer(inputSize, outputSize, activationType));
    }


    public double[] Predict(double[] input)
    {
        double[] output = input;
        foreach (var layer in layers)
            output = layer.Forward(output);
        return output;
    }

    public double Train(double[] input, double[] target)
    {
        double[] output = Predict(input);

        // Вычисление ошибки на выходе
        double[] error = new double[output.Length];
        double totalError = 0;
        for (int i = 0; i < output.Length; i++)
        {
            error[i] = target[i] - output[i];
            totalError += error[i] * error[i];
        }

        // Обратное распространение
        for (int i = layers.Count - 1; i >= 0; i--)
        {
            if (i == layers.Count - 1)
            {
                // Выходной слой: ошибка = target - output
                layers[i].Backward(error, learningRate);
            }
            else
            {
                var nextLayer = layers[i + 1];
                var errors = Utils.ComputeErrors(nextLayer.Weights, nextLayer.Deltas);
                layers[i].Backward(errors, learningRate);
            }
        }

        return totalError;
    }
}
