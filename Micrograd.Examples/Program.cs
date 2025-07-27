using System;
using System.Collections.Generic;
using System.Linq;
using Micrograd.Core;

namespace Micrograd.Examples
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("=== Micrograd .NET Examples ===");
            Console.WriteLine();

            // Example 1: Basic automatic differentiation
            BasicAutodiffExample();
            Console.WriteLine();

            // Example 2: Neural network building blocks
            NeuronLayerExample();
            Console.WriteLine();

            // Example 3: Simple function approximation
            FunctionApproximationExample();
            Console.WriteLine();

            // Example 4: XOR problem
            XORExample();
            Console.WriteLine();

            Console.WriteLine("All examples completed!");
        }

        static void BasicAutodiffExample()
        {
            Console.WriteLine("--- Basic Automatic Differentiation Example ---");
            
            // Create some values
            var x = new Value(2.0, label: "x");
            var y = new Value(3.0, label: "y");
            
            // Build a computational graph: f = x² + 2xy + y²
            var x2 = x * x;
            x2.Label = "x²";
            
            var xy = x * y;
            xy.Label = "xy";
            
            var twoXY = 2.0 * xy;
            twoXY.Label = "2xy";
            
            var y2 = y * y;
            y2.Label = "y²";
            
            var f = x2 + twoXY + y2;
            f.Label = "f";
            
            Console.WriteLine($"Forward pass: f = x² + 2xy + y² = {f.Data}");
            
            // Compute gradients
            f.Backward();
            
            Console.WriteLine($"∂f/∂x = {x.Grad} (expected: 2x + 2y = {2*x.Data + 2*y.Data})");
            Console.WriteLine($"∂f/∂y = {y.Grad} (expected: 2x + 2y = {2*x.Data + 2*y.Data})");
        }

        static void NeuronLayerExample()
        {
            Console.WriteLine("--- Neural Network Building Blocks Example ---");
            
            // Create a single neuron
            var neuron = new Neuron(2, nonlin: true);
            Console.WriteLine($"Created neuron: {neuron}");
            
            // Test the neuron
            var inputs = new[] { new Value(0.5), new Value(-0.3) };
            var output = neuron.Forward(inputs);
            Console.WriteLine($"Neuron output: {output.Data:F4}");
            
            // Create a layer
            var layer = new Layer(2, 3);
            Console.WriteLine($"\nCreated layer: {layer}");
            
            var layerOutputs = layer.Forward(inputs);
            Console.WriteLine($"Layer outputs: [{string.Join(", ", layerOutputs.Select(o => o.Data.ToString("F4")))}]");
            
            // Create an MLP
            var mlp = new MLP(2, new[] { 4, 3, 1 });
            Console.WriteLine($"\nCreated MLP: {mlp}");
            
            var mlpOutput = mlp.ForwardSingle(inputs);
            Console.WriteLine($"MLP output: {mlpOutput.Data:F4}");
            Console.WriteLine($"MLP has {mlp.Parameters().Count()} parameters");
        }

        static void FunctionApproximationExample()
        {
            Console.WriteLine("--- Function Approximation Example ---");
            Console.WriteLine("Learning f(x) = x² using a simple neural network");
            
            // Create a simple network
            var mlp = new MLP(1, new[] { 8, 1 });
            var optimizer = new SGDOptimizer(0.01);
            
            // Generate training data for f(x) = x²
            var trainingData = new List<(Value[] input, Value target)>();
            var random = new Random(42); // Fixed seed for reproducibility
            
            for (int i = 0; i < 100; i++)
            {
                var x = (random.NextDouble() - 0.5) * 4; // Random x in [-2, 2]
                var input = new[] { new Value(x) };
                var target = new Value(x * x);
                trainingData.Add((input, target));
            }
            
            Console.WriteLine($"Generated {trainingData.Count} training samples");
            
            // Training loop
            for (int epoch = 0; epoch < 100; epoch++)
            {
                var totalLoss = 0.0;
                
                foreach (var (input, target) in trainingData)
                {
                    // Forward pass
                    var prediction = mlp.ForwardSingle(input);
                    var loss = LossFunctions.MeanSquaredError(prediction, target);
                    
                    // Backward pass
                    mlp.ZeroGrad();
                    loss.Backward();
                    
                    // Update parameters
                    optimizer.Step(mlp.Parameters());
                    
                    totalLoss += loss.Data;
                }
                
                var avgLoss = totalLoss / trainingData.Count;
                
                if (epoch % 20 == 0)
                {
                    Console.WriteLine($"Epoch {epoch}: Average Loss = {avgLoss:F6}");
                }
            }
            
            // Test the trained network
            Console.WriteLine("\nTesting trained network:");
            var testValues = new[] { -1.5, -1.0, 0.0, 1.0, 1.5 };
            
            foreach (var x in testValues)
            {
                var input = new[] { new Value(x) };
                var prediction = mlp.ForwardSingle(input);
                var expected = x * x;
                Console.WriteLine($"f({x:F1}) = {prediction.Data:F4}, expected = {expected:F4}, error = {Math.Abs(prediction.Data - expected):F4}");
            }
        }

        static void XORExample()
        {
            Console.WriteLine("--- XOR Problem Example ---");
            Console.WriteLine("Learning the XOR function using a neural network");
            
            // Create a network capable of learning XOR (need hidden layer)
            var mlp = new MLP(2, new[] { 4, 1 });
            var optimizer = new SGDOptimizer(0.1);
            
            // XOR training data
            var xorData = new[]
            {
                (new[] { new Value(0.0), new Value(0.0) }, new Value(0.0)),
                (new[] { new Value(0.0), new Value(1.0) }, new Value(1.0)),
                (new[] { new Value(1.0), new Value(0.0) }, new Value(1.0)),
                (new[] { new Value(1.0), new Value(1.0) }, new Value(0.0))
            };
            
            Console.WriteLine("XOR Truth Table:");
            Console.WriteLine("0 XOR 0 = 0");
            Console.WriteLine("0 XOR 1 = 1");
            Console.WriteLine("1 XOR 0 = 1");
            Console.WriteLine("1 XOR 1 = 0");
            Console.WriteLine();
            
            // Training loop
            for (int epoch = 0; epoch < 500; epoch++)
            {
                var totalLoss = 0.0;
                
                foreach (var (input, target) in xorData)
                {
                    // Forward pass
                    var prediction = mlp.ForwardSingle(input);
                    var loss = LossFunctions.MeanSquaredError(prediction, target);
                    
                    // Backward pass
                    mlp.ZeroGrad();
                    loss.Backward();
                    
                    // Update parameters
                    optimizer.Step(mlp.Parameters());
                    
                    totalLoss += loss.Data;
                }
                
                var avgLoss = totalLoss / xorData.Length;
                
                if (epoch % 100 == 0)
                {
                    Console.WriteLine($"Epoch {epoch}: Average Loss = {avgLoss:F6}");
                }
            }
            
            // Test the trained network
            Console.WriteLine("\nTrained Network Results:");
            foreach (var (input, expected) in xorData)
            {
                var prediction = mlp.ForwardSingle(input);
                var x1 = input[0].Data;
                var x2 = input[1].Data;
                var pred = prediction.Data;
                var target = expected.Data;
                
                Console.WriteLine($"{x1:F0} XOR {x2:F0} = {pred:F4} (expected {target:F0})");
            }
            
            // Calculate accuracy
            var correct = 0;
            foreach (var (input, expected) in xorData)
            {
                var prediction = mlp.ForwardSingle(input);
                var predicted = prediction.Data > 0.5 ? 1.0 : 0.0;
                if (Math.Abs(predicted - expected.Data) < 0.01)
                    correct++;
            }
            
            Console.WriteLine($"\nAccuracy: {correct}/{xorData.Length} ({100.0 * correct / xorData.Length:F1}%)");
        }
    }
}
