public class Layer
{
    public int InputSize, OutputSize;
    public double[,] Weights;
    public double[] Biases;
    public double[] Outputs;
    public double[] Inputs;
    public double[] Deltas;
    public Func<double, double> Activation;
    public Func<double, double> ActivationDerivative;

    public ActivationType ActivationKind { get; private set; }

    public Layer(int inputSize, int outputSize, ActivationType activationType = ActivationType.Sigmoid)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
        ActivationKind = activationType;
        Weights = new double[outputSize, inputSize];
        Biases = new double[outputSize];
        Outputs = new double[outputSize];
        Inputs = new double[inputSize];
        Deltas = new double[outputSize];

        switch (activationType)
        {
            case ActivationType.Sigmoid:
                Activation = Sigmoid;
                ActivationDerivative = SigmoidDerivative;
                break;
            case ActivationType.Tanh:
                Activation = Math.Tanh;
                ActivationDerivative = x => 1 - x * x;
                break;
            case ActivationType.ReLU:
                Activation = x => Math.Max(0, x);
                ActivationDerivative = x => x > 0 ? 1 : 0;
                break;
            case ActivationType.Linear:
                Activation = x => x;
                ActivationDerivative = x => 1;
                break;
        }

        Random rand = new Random();
        for (int i = 0; i < outputSize; i++)
        {
            Biases[i] = rand.NextDouble() * 2 - 1;
            for (int j = 0; j < inputSize; j++)
                Weights[i, j] = rand.NextDouble() * 2 - 1;
        }
    }

    public double[] Forward(double[] input)
    {
        Inputs = input;
        for (int i = 0; i < OutputSize; i++)
        {
            double sum = Biases[i];
            for (int j = 0; j < InputSize; j++)
                sum += Weights[i, j] * input[j];
            Outputs[i] = Activation(sum);
        }
        return Outputs;
    }

    public void Backward(double[] errors, double learningRate)
    {
        for (int i = 0; i < errors.Length; i++)
        {
            Deltas[i] = errors[i] * ActivationDerivative(Outputs[i]);
            for (int j = 0; j < InputSize; j++)
                Weights[i, j] += learningRate * Deltas[i] * Inputs[j];
            Biases[i] += learningRate * Deltas[i]; // обновление смещения
        }
    }

    private double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
    private double SigmoidDerivative(double x) => x * (1 - x);
}

public enum ActivationType
{
    Sigmoid,
    Tanh,
    ReLU,
    Linear
}