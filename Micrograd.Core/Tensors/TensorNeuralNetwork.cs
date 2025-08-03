using System;
using System.Collections.Generic;
using System.Linq;
using Micrograd.Core.Backends;

namespace Micrograd.Core
{
    public class TensorNeuron : IDisposable
    {
        public List<TensorValue> Weights { get; private set; }
        public TensorValue Bias { get; private set; }
        public bool NonLinear { get; private set; }
        
        private static readonly Random _random = new Random();

        public TensorNeuron(int nin, bool nonlin = true, ITensorBackend? backend = null)
        {
            backend ??= new GpuBackend();
            
            var weightData = Enumerable.Range(0, nin)
                .Select(_ => (float)(_random.NextDouble() * 2 - 1))
                .ToArray();
            
            Weights = new List<TensorValue>();
            for (int i = 0; i < nin; i++)
            {
                var weightTensor = backend.CreateTensor(new Shape(1), new[] { weightData[i] });
                Weights.Add(new TensorValue(weightTensor));
            }
            
            var biasTensor = backend.CreateTensor(new Shape(1), new[] { (float)(_random.NextDouble() * 2 - 1) });
            Bias = new TensorValue(biasTensor);
            NonLinear = nonlin;
        }

        public TensorValue Forward(IEnumerable<TensorValue> x)
        {
            var inputs = x.ToList();
            if (inputs.Count != Weights.Count)
                throw new ArgumentException($"Expected {Weights.Count} inputs, got {inputs.Count}");

            var sum = Bias;
            for (int i = 0; i < inputs.Count; i++)
            {
                sum = sum + Weights[i] * inputs[i];
            }

            return NonLinear ? sum.Tanh() : sum;
        }

        public IEnumerable<TensorValue> Parameters()
        {
            return Weights.Concat(new[] { Bias });
        }

        public void Dispose()
        {
            foreach (var weight in Weights)
                weight.Dispose();
            Bias?.Dispose();
        }

        public override string ToString()
        {
            var activation = NonLinear ? "Tanh" : "Linear";
            return $"{activation}TensorNeuron({Weights.Count})";
        }
    }

    public class TensorLayer : IDisposable
    {
        public List<TensorNeuron> Neurons { get; private set; }

        public TensorLayer(int nin, int nout, bool nonlin = true, ITensorBackend? backend = null)
        {
            Neurons = Enumerable.Range(0, nout)
                .Select(_ => new TensorNeuron(nin, nonlin, backend))
                .ToList();
        }

        public List<TensorValue> Forward(IEnumerable<TensorValue> x)
        {
            var outputs = Neurons.Select(neuron => neuron.Forward(x)).ToList();
            return outputs;
        }

        public IEnumerable<TensorValue> Parameters()
        {
            return Neurons.SelectMany(neuron => neuron.Parameters());
        }

        public void Dispose()
        {
            foreach (var neuron in Neurons)
                neuron.Dispose();
        }

        public override string ToString()
        {
            var neuronStrings = string.Join(", ", Neurons.Select(n => n.ToString()));
            return $"TensorLayer of [{neuronStrings}]";
        }
    }

    public class TensorMLP : IDisposable
    {
        public List<TensorLayer> Layers { get; private set; }
        public ITensorBackend Backend { get; private set; }

        public TensorMLP(int nin, IEnumerable<int> nouts, ITensorBackend? backend = null)
        {
            Backend = backend ?? new GpuBackend();
            var sizes = new[] { nin }.Concat(nouts).ToList();
            Layers = new List<TensorLayer>();

            for (int i = 0; i < sizes.Count - 1; i++)
            {
                bool isLastLayer = i == sizes.Count - 2;
                Layers.Add(new TensorLayer(sizes[i], sizes[i + 1], !isLastLayer, Backend));
            }
        }

        public IEnumerable<TensorValue> Forward(IEnumerable<TensorValue> x)
        {
            var current = x.ToList();
            
            foreach (var layer in Layers)
            {
                current = layer.Forward(current);
            }

            return current;
        }

        public TensorValue ForwardSingle(IEnumerable<TensorValue> x)
        {
            var outputs = Forward(x).ToList();
            if (outputs.Count != 1)
                throw new InvalidOperationException($"Expected single output, got {outputs.Count}");
            return outputs[0];
        }

        public IEnumerable<TensorValue> Parameters()
        {
            return Layers.SelectMany(layer => layer.Parameters());
        }

        public void ZeroGrad()
        {
            foreach (var param in Parameters())
            {
                param.ZeroGrad();
            }
        }

        public void Dispose()
        {
            foreach (var layer in Layers)
                layer.Dispose();
            Backend?.Dispose();
        }

        public override string ToString()
        {
            var layerStrings = string.Join("\n  ", Layers.Select((layer, i) => $"Layer {i}: {layer}"));
            return $"TensorMLP[\n  {layerStrings}\n]";
        }
    }

    public static class TensorLossFunctions
    {
        public static TensorValue MeanSquaredError(IEnumerable<TensorValue> predictions, IEnumerable<TensorValue> targets)
        {
            var predList = predictions.ToList();
            var targetList = targets.ToList();
            
            if (predList.Count != targetList.Count)
                throw new ArgumentException("Predictions and targets must have the same length");

            var backend = predList[0].Data.Backend;
            var totalLoss = new TensorValue(backend.CreateTensor(new Shape(1), new[] { 0.0f }));
            
            for (int i = 0; i < predList.Count; i++)
            {
                var diff = predList[i] + (targetList[i] * new TensorValue(backend.CreateTensor(new Shape(1), new[] { -1.0f })));
                totalLoss = totalLoss + (diff * diff);
            }

            var countTensor = backend.CreateTensor(new Shape(1), new[] { (float)predList.Count });
            return totalLoss * new TensorValue(backend.CreateTensor(new Shape(1), new[] { 1.0f / predList.Count }));
        }

        public static TensorValue MeanSquaredError(TensorValue prediction, TensorValue target)
        {
            var backend = prediction.Data.Backend;
            var diff = prediction + (target * new TensorValue(backend.CreateTensor(new Shape(1), new[] { -1.0f })));
            return diff * diff;
        }
    }

    public class TensorSGDOptimizer
    {
        public float LearningRate { get; set; }

        public TensorSGDOptimizer(float learningRate = 0.01f)
        {
            LearningRate = learningRate;
        }

        public void Step(IEnumerable<TensorValue> parameters)
        {
            foreach (var param in parameters)
            {
                if (param.Grad.DeviceData != null)
                {
                    var gradData = param.Grad.ToHost();
                    param.Data.Backend.UpdateInPlace(param.Data, gradData, LearningRate);
                }
            }
        }
    }
}