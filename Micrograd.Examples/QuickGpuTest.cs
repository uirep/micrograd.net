using System;
using System.Diagnostics;
using System.Linq;
using Micrograd.Core;
using Micrograd.Core.Backends;

namespace Micrograd.Examples
{
    public static class QuickGpuTest
    {
        public static void RunMatrixBenchmark()
        {
            Console.WriteLine("🚀 QUICK GPU MATRIX BENCHMARK");
            Console.WriteLine("Testing NVIDIA A10 vs CPU performance");
            Console.WriteLine();

            var sizes = new[] { 256, 512, 1024, 1536, 2048, 3072 };
            
            foreach (var size in sizes)
            {
                Console.WriteLine($"--- Matrix {size}×{size} Multiplication ---");
                
                // GPU Test
                ITensorBackend gpuBackend = null;
                TimeSpan gpuTime = TimeSpan.Zero;
                bool gpuSuccess = false;
                
                try
                {
                    gpuBackend = new GpuBackend();
                    gpuTime = BenchmarkMatrixMultiplication(gpuBackend, size);
                    gpuSuccess = true;
                    Console.WriteLine($"🔥 GPU Time: {gpuTime.TotalMilliseconds:F2}ms");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"❌ GPU Failed: {ex.Message}");
                }
                finally
                {
                    gpuBackend?.Dispose();
                }

                // CPU Test (only for smaller sizes to save time)
                if (size <= 1024)
                {
                    using var cpuBackend = new CpuBackend();
                    var cpuTime = BenchmarkMatrixMultiplication(cpuBackend, size);
                    Console.WriteLine($"🖥️  CPU Time: {cpuTime.TotalMilliseconds:F2}ms");

                    if (gpuSuccess && cpuTime > TimeSpan.Zero)
                    {
                        var speedup = cpuTime.TotalMilliseconds / gpuTime.TotalMilliseconds;
                        Console.WriteLine($"🚀 GPU Speedup: {speedup:F2}x faster!");
                    }
                }
                else if (gpuSuccess)
                {
                    Console.WriteLine($"🖥️  CPU skipped (too slow for {size}×{size})");
                    Console.WriteLine($"🚀 GPU handling {size*size:N0} operations in {gpuTime.TotalMilliseconds:F2}ms");
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
            
            // Perform matrix multiplication
            var result = backend.MatMul(a, b);
            // Force computation to complete
            var _ = backend.ToHost(result);
            
            stopwatch.Stop();
            return stopwatch.Elapsed;
        }
    }
}