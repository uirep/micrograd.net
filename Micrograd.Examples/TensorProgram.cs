using System;
using System.Collections.Generic;
using System.Linq;
using Micrograd.Core;
using Micrograd.Core.Backends;

namespace Micrograd.Examples
{
    class TensorProgram
    {
        public static void RunTensorExamples()
        {
            Console.WriteLine("=== Micrograd .NET Tensor Examples ===");
            Console.WriteLine();

            ITensorBackend backend = null;
            
            try
            {
                Console.WriteLine("Initializing GPU backend...");
                backend = new GpuBackend();
                Console.WriteLine("GPU backend initialized successfully!");
                
                RunTensorExamplesWithBackend(backend);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"GPU backend failed: {ex.Message}");
                Console.WriteLine("Falling back to CPU backend...");
                
                backend?.Dispose();
                backend = new CpuBackend();
                Console.WriteLine("CPU backend initialized successfully!");
                
                RunTensorExamplesWithBackend(backend);
            }
            finally
            {
                backend?.Dispose();
            }

            Console.WriteLine("All tensor examples completed!");
        }

        static void RunTensorExamplesWithBackend(ITensorBackend backend)
        {
            TensorBasicExample(backend);
            Console.WriteLine();

            TensorNeuralNetworkExample(backend);
            Console.WriteLine();

            TensorXORExample(backend);
            Console.WriteLine();
        }

        static void TensorBasicExample(ITensorBackend backend)
        {
            Console.WriteLine("--- Tensor Basic Operations Example ---");
            
            var x = new TensorValue(backend.CreateTensor(new Shape(1), new[] { 2.0f }), label: "x");
            var y = new TensorValue(backend.CreateTensor(new Shape(1), new[] { 3.0f }), label: "y");
            
            var result = x * y + x;
            result.Label = "result";
            
            Console.WriteLine($"Forward pass: result = {result.Data.ToHost()[0]:F4}");
            
            result.Backward();
            
            Console.WriteLine($"∂result/∂x = {x.Grad.ToHost()[0]:F4}");
            Console.WriteLine($"∂result/∂y = {y.Grad.ToHost()[0]:F4}");
            
            x.Dispose();
            y.Dispose();
            result.Dispose();
        }

        static void TensorNeuralNetworkExample(ITensorBackend backend)
        {
            Console.WriteLine("--- Tensor Neural Network Example ---");
            
            var mlp = new TensorMLP(2, new[] { 4, 1 }, backend);
            Console.WriteLine($"Created tensor MLP with {mlp.Parameters().Count()} parameters");
            
            var inputs = new[]
            {
                new TensorValue(backend.CreateTensor(new Shape(1), new[] { 0.5f })),
                new TensorValue(backend.CreateTensor(new Shape(1), new[] { -0.3f }))
            };
            
            var output = mlp.ForwardSingle(inputs);
            Console.WriteLine($"MLP output: {output.Data.ToHost()[0]:F4}");
            
            foreach (var input in inputs)
                input.Dispose();
            output.Dispose();
            mlp.Dispose();
        }

        static void TensorXORExample(ITensorBackend backend)
        {
            Console.WriteLine("--- Tensor XOR Problem Example ---");
            
            var mlp = new TensorMLP(2, new[] { 4, 1 }, backend);
            var optimizer = new TensorSGDOptimizer(0.1f);
            
            var xorData = new[]
            {
                (new[] { 0.0f, 0.0f }, 0.0f),
                (new[] { 0.0f, 1.0f }, 1.0f),
                (new[] { 1.0f, 0.0f }, 1.0f),
                (new[] { 1.0f, 1.0f }, 0.0f)
            };
            
            Console.WriteLine("Training XOR function...");
            
            for (int epoch = 0; epoch < 200; epoch++)
            {
                var totalLoss = 0.0f;
                
                foreach (var (inputData, targetData) in xorData)
                {
                    var inputs = inputData.Select(x => new TensorValue(backend.CreateTensor(new Shape(1), new[] { x }))).ToArray();
                    var target = new TensorValue(backend.CreateTensor(new Shape(1), new[] { targetData }));
                    
                    var prediction = mlp.ForwardSingle(inputs);
                    var loss = TensorLossFunctions.MeanSquaredError(prediction, target);
                    
                    mlp.ZeroGrad();
                    loss.Backward();
                    optimizer.Step(mlp.Parameters());
                    
                    totalLoss += loss.Data.ToHost()[0];
                    
                    foreach (var input in inputs)
                        input.Dispose();
                    target.Dispose();
                    prediction.Dispose();
                    loss.Dispose();
                }
                
                if (epoch % 50 == 0)
                {
                    Console.WriteLine($"Epoch {epoch}: Average Loss = {totalLoss / xorData.Length:F6}");
                }
            }
            
            Console.WriteLine("\nTrained Network Results:");
            foreach (var (inputData, expected) in xorData)
            {
                var inputs = inputData.Select(x => new TensorValue(backend.CreateTensor(new Shape(1), new[] { x }))).ToArray();
                var prediction = mlp.ForwardSingle(inputs);
                var pred = prediction.Data.ToHost()[0];
                
                Console.WriteLine($"{inputData[0]:F0} XOR {inputData[1]:F0} = {pred:F4} (expected {expected:F0})");
                
                foreach (var input in inputs)
                    input.Dispose();
                prediction.Dispose();
            }
            
            mlp.Dispose();
        }
    }
}