using System;
using System.Collections.Generic;
using System.Linq;

namespace Micrograd.Core
{
    /// <summary>
    /// A single neuron in a neural network
    /// </summary>
    public class Neuron
    {
        /// <summary>
        /// The weights of the neuron (one for each input)
        /// </summary>
        public List<Value> Weights { get; private set; }

        /// <summary>
        /// The bias of the neuron
        /// </summary>
        public Value Bias { get; private set; }

        /// <summary>
        /// Whether to apply a nonlinear activation function
        /// </summary>
        public bool NonLinear { get; private set; }

        /// <summary>
        /// Random number generator for weight initialization
        /// </summary>
        private static readonly Random _random = new Random();

        /// <summary>
        /// Creates a new neuron with the specified number of inputs
        /// </summary>
        /// <param name="nin">Number of inputs</param>
        /// <param name="nonlin">Whether to apply nonlinear activation</param>
        public Neuron(int nin, bool nonlin = true)
        {
            // Initialize weights with small random values
            Weights = Enumerable.Range(0, nin)
                .Select(_ => new Value(_random.NextDouble() * 2 - 1)) // Random between -1 and 1
                .ToList();
            
            Bias = new Value(_random.NextDouble() * 2 - 1);
            NonLinear = nonlin;
        }

        /// <summary>
        /// Forward pass through the neuron
        /// </summary>
        /// <param name="x">Input values</param>
        /// <returns>Output of the neuron</returns>
        public Value Forward(IEnumerable<Value> x)
        {
            var inputs = x.ToList();
            if (inputs.Count != Weights.Count)
                throw new ArgumentException($"Expected {Weights.Count} inputs, got {inputs.Count}");

            // Compute weighted sum: w*x + b
            var sum = Bias;
            for (int i = 0; i < inputs.Count; i++)
            {
                sum = sum + Weights[i] * inputs[i];
            }

            // Apply activation function if nonlinear
            return NonLinear ? sum.Tanh() : sum;
        }

        /// <summary>
        /// Gets all parameters (weights and bias) of the neuron
        /// </summary>
        /// <returns>All parameters</returns>
        public IEnumerable<Value> Parameters()
        {
            return Weights.Concat(new[] { Bias });
        }

        /// <summary>
        /// String representation of the neuron
        /// </summary>
        public override string ToString()
        {
            var activation = NonLinear ? "Tanh" : "Linear";
            return $"{activation}Neuron({Weights.Count})";
        }
    }

    /// <summary>
    /// A layer of neurons in a neural network
    /// </summary>
    public class Layer
    {
        /// <summary>
        /// The neurons in this layer
        /// </summary>
        public List<Neuron> Neurons { get; private set; }

        /// <summary>
        /// Creates a new layer with the specified dimensions
        /// </summary>
        /// <param name="nin">Number of inputs to each neuron</param>
        /// <param name="nout">Number of neurons in the layer</param>
        /// <param name="nonlin">Whether neurons should use nonlinear activation</param>
        public Layer(int nin, int nout, bool nonlin = true)
        {
            Neurons = Enumerable.Range(0, nout)
                .Select(_ => new Neuron(nin, nonlin))
                .ToList();
        }

        /// <summary>
        /// Forward pass through the layer
        /// </summary>
        /// <param name="x">Input values</param>
        /// <returns>Output values from all neurons</returns>
        public List<Value> Forward(IEnumerable<Value> x)
        {
            var outputs = Neurons.Select(neuron => neuron.Forward(x)).ToList();
            return outputs;
        }

        /// <summary>
        /// Gets all parameters of all neurons in the layer
        /// </summary>
        /// <returns>All parameters</returns>
        public IEnumerable<Value> Parameters()
        {
            return Neurons.SelectMany(neuron => neuron.Parameters());
        }

        /// <summary>
        /// String representation of the layer
        /// </summary>
        public override string ToString()
        {
            var neuronStrings = string.Join(", ", Neurons.Select(n => n.ToString()));
            return $"Layer of [{neuronStrings}]";
        }
    }

    /// <summary>
    /// Multi-Layer Perceptron (MLP) - a complete neural network
    /// </summary>
    public class MLP
    {
        /// <summary>
        /// The layers in the neural network
        /// </summary>
        public List<Layer> Layers { get; private set; }

        /// <summary>
        /// Creates a new MLP with the specified architecture
        /// </summary>
        /// <param name="nin">Number of inputs</param>
        /// <param name="nouts">List of layer sizes (number of neurons in each layer)</param>
        public MLP(int nin, IEnumerable<int> nouts)
        {
            var sizes = new[] { nin }.Concat(nouts).ToList();
            Layers = new List<Layer>();

            for (int i = 0; i < sizes.Count - 1; i++)
            {
                // Last layer is linear (no activation) for regression, all others are nonlinear
                bool isLastLayer = i == sizes.Count - 2;
                Layers.Add(new Layer(sizes[i], sizes[i + 1], !isLastLayer));
            }
        }

        /// <summary>
        /// Forward pass through the entire network
        /// </summary>
        /// <param name="x">Input values</param>
        /// <returns>Output values</returns>
        public IEnumerable<Value> Forward(IEnumerable<Value> x)
        {
            var current = x.ToList();
            
            foreach (var layer in Layers)
            {
                current = layer.Forward(current);
            }

            return current;
        }

        /// <summary>
        /// Convenience method for single output networks
        /// </summary>
        /// <param name="x">Input values</param>
        /// <returns>Single output value</returns>
        public Value ForwardSingle(IEnumerable<Value> x)
        {
            var outputs = Forward(x).ToList();
            if (outputs.Count != 1)
                throw new InvalidOperationException($"Expected single output, got {outputs.Count}");
            return outputs[0];
        }

        /// <summary>
        /// Gets all parameters of the entire network
        /// </summary>
        /// <returns>All parameters</returns>
        public IEnumerable<Value> Parameters()
        {
            return Layers.SelectMany(layer => layer.Parameters());
        }

        /// <summary>
        /// Zeros all gradients in the network
        /// </summary>
        public void ZeroGrad()
        {
            foreach (var param in Parameters())
            {
                param.Grad = 0.0;
            }
        }

        /// <summary>
        /// String representation of the MLP
        /// </summary>
        public override string ToString()
        {
            var layerStrings = string.Join("\n  ", Layers.Select((layer, i) => $"Layer {i}: {layer}"));
            return $"MLP[\n  {layerStrings}\n]";
        }
    }

    /// <summary>
    /// Common loss functions for training neural networks
    /// </summary>
    public static class LossFunctions
    {
        /// <summary>
        /// Mean Squared Error loss function
        /// </summary>
        /// <param name="predictions">Predicted values</param>
        /// <param name="targets">Target values</param>
        /// <returns>MSE loss</returns>
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

        /// <summary>
        /// Mean Squared Error loss for single values
        /// </summary>
        /// <param name="prediction">Predicted value</param>
        /// <param name="target">Target value</param>
        /// <returns>MSE loss</returns>
        public static Value MeanSquaredError(Value prediction, Value target)
        {
            var diff = prediction - target;
            return diff * diff;
        }
    }

    /// <summary>
    /// Simple gradient descent optimizer
    /// </summary>
    public class SGDOptimizer
    {
        /// <summary>
        /// Learning rate for gradient descent
        /// </summary>
        public double LearningRate { get; set; }

        /// <summary>
        /// Creates a new SGD optimizer
        /// </summary>
        /// <param name="learningRate">Learning rate</param>
        public SGDOptimizer(double learningRate = 0.01)
        {
            LearningRate = learningRate;
        }

        /// <summary>
        /// Performs one step of gradient descent
        /// </summary>
        /// <param name="parameters">Parameters to update</param>
        public void Step(IEnumerable<Value> parameters)
        {
            foreach (var param in parameters)
            {
                param.Data -= LearningRate * param.Grad;
            }
        }
    }
} 