using Micrograd.Core;
using Micrograd.Core.Backends;

namespace Micrograd.Tests
{
    public class LegacyCompatibilityTests : IDisposable
    {
        private readonly ITensorBackend _backend;
        private static readonly Random rand = new();

        public LegacyCompatibilityTests()
        {
            try
            {
                _backend = new GpuBackend();
            }
            catch
            {
                _backend = new CpuBackend();
            }
        }

        [Fact]
        public void TensorAdd_WorksCorrectly()
        {
            var a = _backend.CreateTensor(new Shape(3), new float[] { 1, 2, 3 });
            var b = _backend.CreateTensor(new Shape(3), new float[] { 4, 5, 6 });
            var c = a + b;
            Assert.Equal(new float[] { 5, 7, 9 }, c.ToHost());
            
            a.Dispose();
            b.Dispose();
            c.Dispose();
        }

        [Fact]
        public void TensorAdd_WorksCorrectly_WithLargeVectors()
        {
            var a = _backend.CreateTensor(new Shape(1000000), new float[1000000]);
            var b = _backend.CreateTensor(new Shape(1000000), new float[1000000]);
            var c = a + b;
            Assert.Equal(new float[1000000], c.ToHost());
            
            a.Dispose();
            b.Dispose();
            c.Dispose();
        }

        [Fact]
        public void TensorAdd_MatchesCpuVersion()
        {
            var data1 = Enumerable.Range(0, 100000).Select(_ => (float)rand.NextDouble()).ToArray();
            var data2 = data1.Select(x => x * 0.5f).ToArray();
            
            var gpuA = _backend.CreateTensor(new Shape(100000), data1);
            var gpuB = _backend.CreateTensor(new Shape(100000), data2);
            var gpuResult = gpuA + gpuB;
            
            var cpuBackend = new CpuBackend();
            var cpuA = cpuBackend.CreateTensor(new Shape(100000), data1);
            var cpuB = cpuBackend.CreateTensor(new Shape(100000), data2);
            var cpuResult = cpuA + cpuB;

            var gpuData = gpuResult.ToHost();
            var cpuData = cpuResult.ToHost();

            for (int i = 0; i < data1.Length; i++)
            {
                Assert.Equal(cpuData[i], gpuData[i], 1e-5);
            }
            
            gpuA.Dispose();
            gpuB.Dispose();
            gpuResult.Dispose();
            cpuA.Dispose();
            cpuB.Dispose();
            cpuResult.Dispose();
            cpuBackend.Dispose();
        }

        [Fact]
        public void TensorAdd_ThrowsOnDifferentShapes()
        {
            var a = _backend.CreateTensor(new Shape(3), new float[] { 1, 2, 3 });
            var b = _backend.CreateTensor(new Shape(4), new float[] { 4, 5, 6, 7 });
            
            Assert.Throws<ArgumentException>(() => a + b);
            
            a.Dispose();
            b.Dispose();
        }

        public void Dispose()
        {
            _backend?.Dispose();
        }
    }
}