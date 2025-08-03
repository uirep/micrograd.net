using System;
using System.Diagnostics;
using System.Linq;
using Micrograd.Core;
using Micrograd.Core.Backends;

namespace Micrograd.Examples
{
    public static class GpuBenchmark
    {
        public static void RunHeavyGpuTest()
        {
            Console.WriteLine("=== HEAVY GPU PERFORMANCE BENCHMARK ===");
            Console.WriteLine($"Testing with NVIDIA GPU vs CPU performance");
            Console.WriteLine();

            // Test 1: Large Matrix Multiplications
            Console.WriteLine("🔥 TEST 1: Large Matrix Multiplications");
            RunMatrixMultiplicationBenchmark();
            Console.WriteLine();

            // Test 2: Deep Neural Network Training
            Console.WriteLine("🔥 TEST 2: Deep Neural Network Training");
            RunDeepNetworkBenchmark();
            Console.WriteLine();

            // Test 3: Massive Tensor Operations
            Console.WriteLine("🔥 TEST 3: Massive Tensor Operations");
            RunMassiveTensorBenchmark();
            Console.WriteLine();

            Console.WriteLine("🎉 GPU BENCHMARK COMPLETED!");
        }

        private static void RunMatrixMultiplicationBenchmark()
        {
            var sizes = new[] { 512, 1024, 2048 };
            
            foreach (var size in sizes)
            {
                Console.WriteLine($"--- Matrix {size}x{size} Multiplication ---");
                
                // GPU Test
                ITensorBackend gpuBackend = null;
                TimeSpan gpuTime = TimeSpan.Zero;
                bool gpuSuccess = false;
                
                try
                {
                    gpuBackend = new GpuBackend();
                    gpuTime = BenchmarkMatrixMultiplication(gpuBackend, size);
                    gpuSuccess = true;
                    Console.WriteLine($"✅ GPU Time: {gpuTime.TotalMilliseconds:F2}ms");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"❌ GPU Failed: {ex.Message}");
                }
                finally
                {
                    gpuBackend?.Dispose();
                }

                // CPU Test
                using var cpuBackend = new CpuBackend();
                var cpuTime = BenchmarkMatrixMultiplication(cpuBackend, size);
                Console.WriteLine($"🖥️  CPU Time: {cpuTime.TotalMilliseconds:F2}ms");

                if (gpuSuccess && cpuTime > TimeSpan.Zero)
                {
                    var speedup = cpuTime.TotalMilliseconds / gpuTime.TotalMilliseconds;
                    Console.WriteLine($"🚀 GPU Speedup: {speedup:F2}x faster");
                }
                Console.WriteLine();
            }
        }

        private static TimeSpan BenchmarkMatrixMultiplication(ITensorBackend backend, int size)
        {
            // Create random matrices
            var random = new Random(42);
            var aData = Enumerable.Range(0, size * size).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
            var bData = Enumerable.Range(0, size * size).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();

            var a = backend.CreateTensor(new Shape(size, size), aData);
            var b = backend.CreateTensor(new Shape(size, size), bData);

            var stopwatch = Stopwatch.StartNew();
            
            // Perform multiple matrix multiplications to stress test
            for (int i = 0; i < 5; i++)
            {
                var result = backend.MatMul(a, b);
                // Force computation to complete
                var _ = backend.ToHost(result);
            }
            
            stopwatch.Stop();
            return stopwatch.Elapsed;
        }

        private static void RunDeepNetworkBenchmark()
        {
            Console.WriteLine("Creating deep neural network with large layers...");
            
            // GPU Test
            ITensorBackend gpuBackend = null;
            TimeSpan gpuTime = TimeSpan.Zero;
            bool gpuSuccess = false;
            
            try
            {
                gpuBackend = new GpuBackend();
                gpuTime = BenchmarkDeepNetwork(gpuBackend);
                gpuSuccess = true;
                Console.WriteLine($"✅ GPU Training Time: {gpuTime.TotalMilliseconds:F2}ms");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ GPU Failed: {ex.Message}");
            }
            finally
            {
                gpuBackend?.Dispose();
            }

            // CPU Test
            using var cpuBackend = new CpuBackend();
            var cpuTime = BenchmarkDeepNetwork(cpuBackend);
            Console.WriteLine($"🖥️  CPU Training Time: {cpuTime.TotalMilliseconds:F2}ms");

            if (gpuSuccess && cpuTime > TimeSpan.Zero)
            {
                var speedup = cpuTime.TotalMilliseconds / gpuTime.TotalMilliseconds;
                Console.WriteLine($"🚀 GPU Speedup: {speedup:F2}x faster");
            }
        }

        private static TimeSpan BenchmarkDeepNetwork(ITensorBackend backend)
        {
            var random = new Random(42);
            
            // Create a deep network: 256 -> 512 -> 512 -> 256 -> 128 -> 1
            var layers = new[]
            {
                CreateDenseLayer(backend, 256, 512),
                CreateDenseLayer(backend, 512, 512), 
                CreateDenseLayer(backend, 512, 256),
                CreateDenseLayer(backend, 256, 128),
                CreateDenseLayer(backend, 128, 1)
            };

            // Generate large batch of training data
            var batchSize = 128;
            var inputData = Enumerable.Range(0, batchSize * 256)
                .Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
            var input = backend.CreateTensor(new Shape(batchSize, 256), inputData);

            var stopwatch = Stopwatch.StartNew();
            
            // Forward pass through deep network multiple times
            for (int epoch = 0; epoch < 10; epoch++)
            {
                var current = input;
                
                // Forward pass through all layers
                foreach (var (weights, biases) in layers)
                {
                    // Linear transformation: current * weights (skip bias for simplicity)
                    current = backend.MatMul(current, weights);
                    current = backend.Tanh(current); // Activation
                }
                
                // Force computation to complete
                var _ = backend.ToHost(current);
            }
            
            stopwatch.Stop();
            return stopwatch.Elapsed;
        }

        private static (Tensor weights, Tensor biases) CreateDenseLayer(ITensorBackend backend, int inputSize, int outputSize)
        {
            var random = new Random(42);
            
            // Xavier initialization
            var scale = Math.Sqrt(2.0 / (inputSize + outputSize));
            var weightsData = Enumerable.Range(0, inputSize * outputSize)
                .Select(_ => (float)(random.NextGaussian() * scale)).ToArray();
            var biasesData = new float[outputSize]; // Initialize to zero
            
            var weights = backend.CreateTensor(new Shape(inputSize, outputSize), weightsData);
            var biases = backend.CreateTensor(new Shape(1, outputSize), biasesData);
            
            return (weights, biases);
        }

        private static void RunMassiveTensorBenchmark()
        {
            Console.WriteLine("Performing massive tensor operations...");
            
            var sizes = new[] { 10000, 50000, 100000 };
            
            foreach (var size in sizes)
            {
                Console.WriteLine($"--- Vector Size: {size:N0} elements ---");
                
                // GPU Test
                ITensorBackend gpuBackend = null;
                TimeSpan gpuTime = TimeSpan.Zero;
                bool gpuSuccess = false;
                
                try
                {
                    gpuBackend = new GpuBackend();
                    gpuTime = BenchmarkMassiveTensorOps(gpuBackend, size);
                    gpuSuccess = true;
                    Console.WriteLine($"✅ GPU Time: {gpuTime.TotalMilliseconds:F2}ms");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"❌ GPU Failed: {ex.Message}");
                }
                finally
                {
                    gpuBackend?.Dispose();
                }

                // CPU Test  
                using var cpuBackend = new CpuBackend();
                var cpuTime = BenchmarkMassiveTensorOps(cpuBackend, size);
                Console.WriteLine($"🖥️  CPU Time: {cpuTime.TotalMilliseconds:F2}ms");

                if (gpuSuccess && cpuTime > TimeSpan.Zero)
                {
                    var speedup = cpuTime.TotalMilliseconds / gpuTime.TotalMilliseconds;
                    Console.WriteLine($"🚀 GPU Speedup: {speedup:F2}x faster");
                }
                Console.WriteLine();
            }
        }

        private static TimeSpan BenchmarkMassiveTensorOps(ITensorBackend backend, int size)
        {
            var random = new Random(42);
            
            // Create large tensors
            var aData = Enumerable.Range(0, size).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
            var bData = Enumerable.Range(0, size).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
            
            var a = backend.CreateTensor(new Shape(size), aData);
            var b = backend.CreateTensor(new Shape(size), bData);

            var stopwatch = Stopwatch.StartNew();
            
            // Chain of operations to stress test
            for (int i = 0; i < 20; i++)
            {
                var result = backend.Add(a, b);          // Addition
                result = backend.Multiply(result, a);    // Multiplication  
                result = backend.Tanh(result);           // Tanh activation
                result = backend.ReLU(result);           // ReLU activation
                
                // Force computation
                var _ = backend.ToHost(result);
            }
            
            stopwatch.Stop();
            return stopwatch.Elapsed;
        }
    }

    // Extension method for Gaussian random numbers
    public static class RandomExtensions
    {
        public static double NextGaussian(this Random random, double mean = 0, double stdDev = 1)
        {
            // Box-Muller transform
            static double Generate(Random rnd)
            {
                double u1 = 1.0 - rnd.NextDouble();
                double u2 = 1.0 - rnd.NextDouble();
                return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            }
            
            return mean + stdDev * Generate(random);
        }
    }
}