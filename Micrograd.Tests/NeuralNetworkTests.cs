using Xunit;
using Micrograd.Core;
using System.Linq;

namespace Micrograd.Tests
{
    public class NeuralNetworkTests
    {
        private const double Tolerance = 1e-6;

        [Fact]
        public void Neuron_Constructor_InitializesCorrectly()
        {
            var neuron = new Neuron(3, nonlin: true);
            
            Assert.Equal(3, neuron.Weights.Count);
            Assert.NotNull(neuron.Bias);
            Assert.True(neuron.NonLinear);
            
            // Check that weights are initialized with reasonable values
            Assert.All(neuron.Weights, w => Assert.True(w.Data >= -1.0 && w.Data <= 1.0));
            Assert.True(neuron.Bias.Data >= -1.0 && neuron.Bias.Data <= 1.0);
        }

        [Fact]
        public void Neuron_Forward_WorksCorrectly()
        {
            var neuron = new Neuron(2, nonlin: false); // Linear neuron for predictable testing
            
            // Set specific weights for testing
            neuron.Weights[0].Data = 0.5;
            neuron.Weights[1].Data = 0.3;
            neuron.Bias.Data = 0.1;
            
            var inputs = new[] { new Value(2.0), new Value(4.0) };
            var output = neuron.Forward(inputs);
            
            // Expected: 0.5*2 + 0.3*4 + 0.1 = 1.0 + 1.2 + 0.1 = 2.3
            Assert.Equal(2.3, output.Data, Tolerance);
        }

        [Fact]
        public void Neuron_Forward_ThrowsOnWrongInputCount()
        {
            var neuron = new Neuron(2);
            var inputs = new[] { new Value(1.0) }; // Wrong number of inputs
            
            Assert.Throws<System.ArgumentException>(() => neuron.Forward(inputs));
        }

        [Fact]
        public void Neuron_Parameters_ReturnsAllParameters()
        {
            var neuron = new Neuron(3);
            var parameters = neuron.Parameters().ToList();
            
            // Should return 3 weights + 1 bias = 4 parameters
            Assert.Equal(4, parameters.Count);
        }

        [Fact]
        public void Layer_Constructor_InitializesCorrectly()
        {
            var layer = new Layer(2, 3); // 2 inputs, 3 neurons
            
            Assert.Equal(3, layer.Neurons.Count);
            Assert.All(layer.Neurons, n => Assert.Equal(2, n.Weights.Count));
        }

        [Fact]
        public void Layer_Forward_WorksCorrectly()
        {
            var layer = new Layer(2, 2, nonlin: false); // 2 inputs, 2 linear neurons
            
            var inputs = new[] { new Value(1.0), new Value(2.0) };
            var outputs = layer.Forward(inputs);
            
            Assert.Equal(2, outputs.Count);
            Assert.All(outputs, output => Assert.NotNull(output));
        }

        [Fact]
        public void Layer_Parameters_ReturnsAllParameters()
        {
            var layer = new Layer(2, 3); // 2 inputs, 3 neurons
            var parameters = layer.Parameters().ToList();
            
            // Each neuron has 2 weights + 1 bias = 3 parameters
            // 3 neurons * 3 parameters = 9 total parameters
            Assert.Equal(9, parameters.Count);
        }

        [Fact]
        public void MLP_Constructor_InitializesCorrectly()
        {
            var mlp = new MLP(3, new[] { 4, 2, 1 }); // 3 inputs, hidden layer of 4, hidden layer of 2, output of 1
            
            Assert.Equal(3, mlp.Layers.Count);
            
            // First layer: 3 inputs -> 4 outputs
            Assert.Equal(4, mlp.Layers[0].Neurons.Count);
            Assert.All(mlp.Layers[0].Neurons, n => Assert.Equal(3, n.Weights.Count));
            
            // Second layer: 4 inputs -> 2 outputs
            Assert.Equal(2, mlp.Layers[1].Neurons.Count);
            Assert.All(mlp.Layers[1].Neurons, n => Assert.Equal(4, n.Weights.Count));
            
            // Third layer: 2 inputs -> 1 output (linear)
            Assert.Equal(1, mlp.Layers[2].Neurons.Count);
            Assert.All(mlp.Layers[2].Neurons, n => Assert.Equal(2, n.Weights.Count));
            Assert.False(mlp.Layers[2].Neurons[0].NonLinear); // Last layer should be linear
        }

        [Fact]
        public void MLP_Forward_WorksCorrectly()
        {
            var mlp = new MLP(2, new[] { 3, 1 }); // 2 inputs, hidden layer of 3, output of 1
            
            var inputs = new[] { new Value(1.0), new Value(2.0) };
            var outputs = mlp.Forward(inputs).ToList();
            
            Assert.Equal(1, outputs.Count);
            Assert.NotNull(outputs[0]);
        }

        [Fact]
        public void MLP_ForwardSingle_WorksCorrectly()
        {
            var mlp = new MLP(2, new[] { 3, 1 }); // Single output network
            
            var inputs = new[] { new Value(1.0), new Value(2.0) };
            var output = mlp.ForwardSingle(inputs);
            
            Assert.NotNull(output);
        }

        [Fact]
        public void MLP_ForwardSingle_ThrowsOnMultipleOutputs()
        {
            var mlp = new MLP(2, new[] { 3, 2 }); // Multiple output network
            
            var inputs = new[] { new Value(1.0), new Value(2.0) };
            
            Assert.Throws<System.InvalidOperationException>(() => mlp.ForwardSingle(inputs));
        }

        [Fact]
        public void MLP_Parameters_ReturnsAllParameters()
        {
            var mlp = new MLP(2, new[] { 2, 1 }); // 2->2->1
            var parameters = mlp.Parameters().ToList();
            
            // First layer: 2 neurons * (2 weights + 1 bias) = 6 parameters
            // Second layer: 1 neuron * (2 weights + 1 bias) = 3 parameters
            // Total: 9 parameters
            Assert.Equal(9, parameters.Count);
        }

        [Fact]
        public void MLP_ZeroGrad_ZerosAllGradients()
        {
            var mlp = new MLP(2, new[] { 2, 1 });
            
            // Set some gradients
            foreach (var param in mlp.Parameters())
            {
                param.Grad = 1.0;
            }
            
            mlp.ZeroGrad();
            
            // Check all gradients are zero
            Assert.All(mlp.Parameters(), param => Assert.Equal(0.0, param.Grad, Tolerance));
        }

        [Fact]
        public void LossFunctions_MeanSquaredError_WorksCorrectly()
        {
            var predictions = new[] { new Value(1.0), new Value(2.0), new Value(3.0) };
            var targets = new[] { new Value(1.1), new Value(2.2), new Value(2.8) };
            
            var loss = LossFunctions.MeanSquaredError(predictions, targets);
            
            // MSE = ((1.0-1.1)^2 + (2.0-2.2)^2 + (3.0-2.8)^2) / 3
            // = (0.01 + 0.04 + 0.04) / 3 = 0.03
            Assert.Equal(0.03, loss.Data, Tolerance);
        }

        [Fact]
        public void LossFunctions_MeanSquaredError_Single_WorksCorrectly()
        {
            var prediction = new Value(2.0);
            var target = new Value(1.5);
            
            var loss = LossFunctions.MeanSquaredError(prediction, target);
            
            // MSE = (2.0 - 1.5)^2 = 0.25
            Assert.Equal(0.25, loss.Data, Tolerance);
        }

        [Fact]
        public void SGDOptimizer_Step_UpdatesParameters()
        {
            var optimizer = new SGDOptimizer(0.1);
            var param = new Value(1.0);
            param.Grad = 0.5;
            
            var originalData = param.Data;
            optimizer.Step(new[] { param });
            
            // New value = old value - learning_rate * gradient = 1.0 - 0.1 * 0.5 = 0.95
            Assert.Equal(0.95, param.Data, Tolerance);
        }

        [Fact]
        public void NeuralNetwork_EndToEnd_TrainingStep()
        {
            // Create a simple network
            var mlp = new MLP(2, new[] { 2, 1 });
            var optimizer = new SGDOptimizer(0.01);
            
            // Sample input and target
            var input = new[] { new Value(0.5), new Value(-0.3) };
            var target = new Value(1.0);
            
            // Forward pass
            var prediction = mlp.ForwardSingle(input);
            var loss = LossFunctions.MeanSquaredError(prediction, target);
            
            // Backward pass
            mlp.ZeroGrad();
            loss.Backward();
            
            // Check that gradients are computed
            Assert.True(mlp.Parameters().Any(p => p.Grad != 0.0));
            
            // Optimization step
            optimizer.Step(mlp.Parameters());
            
            // Parameters should have changed
            Assert.True(mlp.Parameters().Any(p => p.Data != 0.0));
        }
    }
} 