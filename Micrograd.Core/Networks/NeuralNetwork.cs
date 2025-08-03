using System;
using System.Collections.Generic;
using System.Linq;

namespace Micrograd.Core
{

    public class Neuron
    {
        public List<Value> Weights { get; private set; }
        public Value Bias { get; private set; }
        public bool NonLinear { get; private set; }
        
        private static readonly Random _random = new Random();

        public Neuron(int nin, bool nonlin = true)
        {
            Weights = Enumerable.Range(0, nin)
                .Select(_ => new Value(_random.NextDouble() * 2 - 1))
                .ToList();
            
            Bias = new Value(_random.NextDouble() * 2 - 1);
            NonLinear = nonlin;
        }

        public Value Forward(IEnumerable<Value> x)
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

        public IEnumerable<Value> Parameters()
        {
            return Weights.Concat(new[] { Bias });
        }

        public override string ToString()
        {
            var activation = NonLinear ? "Tanh" : "Linear";
            return $"{activation}Neuron({Weights.Count})";
        }
    }

    public class Layer
    {
        public List<Neuron> Neurons { get; private set; }

        public Layer(int nin, int nout, bool nonlin = true)
        {
            Neurons = Enumerable.Range(0, nout)
                .Select(_ => new Neuron(nin, nonlin))
                .ToList();
        }

        public List<Value> Forward(IEnumerable<Value> x)
        {
            var outputs = Neurons.Select(neuron => neuron.Forward(x)).ToList();
            return outputs;
        }

        public IEnumerable<Value> Parameters()
        {
            return Neurons.SelectMany(neuron => neuron.Parameters());
        }

        public override string ToString()
        {
            var neuronStrings = string.Join(", ", Neurons.Select(n => n.ToString()));
            return $"Layer of [{neuronStrings}]";
        }
    }

    public class MLP
    {
        public List<Layer> Layers { get; private set; }

        public MLP(int nin, IEnumerable<int> nouts)
        {
            var sizes = new[] { nin }.Concat(nouts).ToList();
            Layers = new List<Layer>();

            for (int i = 0; i < sizes.Count - 1; i++)
            {
                bool isLastLayer = i == sizes.Count - 2;
                Layers.Add(new Layer(sizes[i], sizes[i + 1], !isLastLayer));
            }
        }

        public IEnumerable<Value> Forward(IEnumerable<Value> x)
        {
            var current = x.ToList();
            
            foreach (var layer in Layers)
            {
                current = layer.Forward(current);
            }

            return current;
        }

        public Value ForwardSingle(IEnumerable<Value> x)
        {
            var outputs = Forward(x).ToList();
            if (outputs.Count != 1)
                throw new InvalidOperationException($"Expected single output, got {outputs.Count}");
            return outputs[0];
        }

        public IEnumerable<Value> Parameters()
        {
            return Layers.SelectMany(layer => layer.Parameters());
        }

        public void ZeroGrad()
        {
            foreach (var param in Parameters())
            {
                param.Grad = 0.0;
            }
        }

        public override string ToString()
        {
            var layerStrings = string.Join("\n  ", Layers.Select((layer, i) => $"Layer {i}: {layer}"));
            return $"MLP[\n  {layerStrings}\n]";
        }
    }
}