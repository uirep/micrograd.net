using System;
using System.Collections.Generic;
using System.Linq;

namespace Micrograd.Core
{
    public static class LossFunctions
    {
        public static Value MeanSquaredError(IEnumerable<Value> predictions, IEnumerable<Value> targets)
        {
            var predList = predictions.ToList();
            var targetList = targets.ToList();
            
            if (predList.Count != targetList.Count)
                throw new ArgumentException("Predictions and targets must have the same length");

            var totalLoss = new Value(0.0);
            for (int i = 0; i < predList.Count; i++)
            {
                var diff = predList[i] - targetList[i];
                totalLoss = totalLoss + diff * diff;
            }

            return totalLoss / (double)predList.Count;
        }

        public static Value MeanSquaredError(Value prediction, Value target)
        {
            var diff = prediction - target;
            return diff * diff;
        }
    }

    public class SGDOptimizer
    {
        public double LearningRate { get; set; }

        public SGDOptimizer(double learningRate = 0.01)
        {
            LearningRate = learningRate;
        }

        public void Step(IEnumerable<Value> parameters)
        {
            foreach (var param in parameters)
            {
                param.Data -= LearningRate * param.Grad;
            }
        }
    }
}