using Xunit;
using Micrograd.Core;
using System;
using System.Linq;

namespace Micrograd.Tests.Networks
{
    public class NeuralNetworkTests
    {
        [Fact]
        public void Neuron_CreatesCorrectly()
        {
            var neuron = new Neuron(3, true);
            
            Assert.Equal(3, neuron.Weights.Count);
            Assert.NotNull(neuron.Bias);
            Assert.True(neuron.NonLinear);
            Assert.Equal(4, neuron.Parameters().Count());
        }

        [Fact]
        public void Neuron_ForwardPass()
        {
            var neuron = new Neuron(2, false);
            var inputs = new[] { new Value(1.0), new Value(2.0) };
            
            var output = neuron.Forward(inputs);
            
            Assert.NotNull(output);
        }

        [Fact]
        public void Neuron_ThrowsOnWrongInputSize()
        {
            var neuron = new Neuron(3);
            var inputs = new[] { new Value(1.0), new Value(2.0) };
            
            Assert.Throws<ArgumentException>(() => neuron.Forward(inputs));
        }

        [Fact]
        public void Layer_CreatesCorrectly()
        {
            var layer = new Layer(3, 4, true);
            
            Assert.Equal(4, layer.Neurons.Count);
            Assert.Equal(16, layer.Parameters().Count());
        }

        [Fact]
        public void Layer_ForwardPass()
        {
            var layer = new Layer(2, 3, false);
            var inputs = new[] { new Value(0.5), new Value(-0.3) };
            
            var outputs = layer.Forward(inputs);
            
            Assert.Equal(3, outputs.Count);
        }

        [Fact]
        public void MLP_CreatesCorrectly()
        {
            var mlp = new MLP(2, new[] { 4, 3, 1 });
            
            Assert.Equal(3, mlp.Layers.Count);
            Assert.Equal(26, mlp.Parameters().Count());
        }

        [Fact]
        public void MLP_ForwardPass()
        {
            var mlp = new MLP(2, new[] { 3, 1 });
            var inputs = new[] { new Value(1.0), new Value(-1.0) };
            
            var output = mlp.ForwardSingle(inputs);
            
            Assert.NotNull(output);
        }

        [Fact]
        public void MLP_MultipleOutputs()
        {
            var mlp = new MLP(2, new[] { 3, 2 });
            var inputs = new[] { new Value(1.0), new Value(-1.0) };
            
            var outputs = mlp.Forward(inputs);
            
            Assert.Equal(2, outputs.Count());
        }

        [Fact]
        public void MLP_ThrowsOnSingleOutputFromMultiple()
        {
            var mlp = new MLP(2, new[] { 3, 2 });
            var inputs = new[] { new Value(1.0), new Value(-1.0) };
            
            Assert.Throws<InvalidOperationException>(() => mlp.ForwardSingle(inputs));
        }

        [Fact]
        public void MLP_TrainingSingleStep()
        {
            var mlp = new MLP(2, new[] { 3, 1 });
            var optimizer = new SGDOptimizer(0.01);
            
            var inputs = new[] { new Value(1.0), new Value(0.0) };
            var target = new Value(1.0);
            
            var prediction = mlp.ForwardSingle(inputs);
            var loss = LossFunctions.MeanSquaredError(prediction, target);
            
            mlp.ZeroGrad();
            loss.Backward();
            optimizer.Step(mlp.Parameters());
            
            Assert.True(loss.Data >= 0);
        }

        [Fact]
        public void MLP_XORLearning()
        {
            var mlp = new MLP(2, new[] { 4, 1 });
            var optimizer = new SGDOptimizer(0.1);
            
            var xorData = new[]
            {
                (new[] { new Value(0.0), new Value(0.0) }, new Value(0.0)),
                (new[] { new Value(0.0), new Value(1.0) }, new Value(1.0)),
                (new[] { new Value(1.0), new Value(0.0) }, new Value(1.0)),
                (new[] { new Value(1.0), new Value(1.0) }, new Value(0.0))
            };
            
            var initialLoss = 0.0;
            var finalLoss = 0.0;
            
            for (int epoch = 0; epoch < 50; epoch++)
            {
                var epochLoss = 0.0;
                
                foreach (var (inputs, target) in xorData)
                {
                    var prediction = mlp.ForwardSingle(inputs);
                    var loss = LossFunctions.MeanSquaredError(prediction, target);
                    
                    mlp.ZeroGrad();
                    loss.Backward();
                    optimizer.Step(mlp.Parameters());
                    
                    epochLoss += loss.Data;
                }
                
                if (epoch == 0) initialLoss = epochLoss;
                if (epoch == 49) finalLoss = epochLoss;
            }
            
            Assert.True(finalLoss < initialLoss, $"Loss should decrease: {initialLoss} -> {finalLoss}");
        }

        [Fact]
        public void MLP_ZeroGrad()
        {
            var mlp = new MLP(2, new[] { 2, 1 });
            var inputs = new[] { new Value(1.0), new Value(-1.0) };
            
            var output = mlp.ForwardSingle(inputs);
            output.Backward();
            
            var hasNonZeroGrads = mlp.Parameters().Any(p => p.Grad != 0.0);
            Assert.True(hasNonZeroGrads);
            
            mlp.ZeroGrad();
            
            var allZeroGrads = mlp.Parameters().All(p => p.Grad == 0.0);
            Assert.True(allZeroGrads);
        }

        [Fact]
        public void MLP_ToString()
        {
            var mlp = new MLP(2, new[] { 3, 1 });
            var str = mlp.ToString();
            
            Assert.Contains("MLP", str);
            Assert.Contains("Layer", str);
        }
    }
}