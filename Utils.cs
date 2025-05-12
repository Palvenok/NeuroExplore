public class Utils
{
    
    public static double[] ComputeErrors(double[,] nextWeights, double[] nextDeltas)
    {
        int currentOutputSize = nextWeights.GetLength(1); // ← количество выходов текущего слоя
        double[] errors = new double[currentOutputSize];

        for (int i = 0; i < currentOutputSize; i++)
        {
            double error = 0;
            for (int j = 0; j < nextDeltas.Length; j++)
                error += nextWeights[j, i] * nextDeltas[j];
            errors[i] = error;
        }
        return errors;
    }
}