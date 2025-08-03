using Xunit;
using Micrograd.Core;
using Micrograd.Core.Backends;

namespace Micrograd.Tests.Tensors
{
    public class TensorTests : IDisposable
    {
        private readonly ITensorBackend _gpuBackend;
        private readonly ITensorBackend _cpuBackend;

        public TensorTests()
        {
            try
            {
                _gpuBackend = new GpuBackend();
            }
            catch
            {
                _gpuBackend = new CpuBackend();
            }
            _cpuBackend = new CpuBackend();
        }

        [Fact]
        public void Shape_CreatesCorrectly()
        {
            var shape = new Shape(2, 3, 4);
            Assert.Equal(3, shape.Rank);
            Assert.Equal(24, shape.Size);
            Assert.Equal(2, shape[0]);
            Assert.Equal(3, shape[1]);
            Assert.Equal(4, shape[2]);
        }

        [Fact]
        public void Shape_EqualityWorks()
        {
            var shape1 = new Shape(2, 3);
            var shape2 = new Shape(2, 3);
            var shape3 = new Shape(3, 2);
            
            Assert.Equal(shape1, shape2);
            Assert.NotEqual(shape1, shape3);
        }

        [Fact]
        public void Tensor_CreateAndToHost()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var tensor = _gpuBackend.CreateTensor(new Shape(3), data);
            var result = tensor.ToHost();
            
            Assert.Equal(data, result);
            tensor.Dispose();
        }

        [Fact]
        public void Tensor_Addition()
        {
            var a = _gpuBackend.CreateTensor(new Shape(3), new float[] { 1, 2, 3 });
            var b = _gpuBackend.CreateTensor(new Shape(3), new float[] { 4, 5, 6 });
            var result = a + b;
            
            Assert.Equal(new float[] { 5, 7, 9 }, result.ToHost());
            
            a.Dispose();
            b.Dispose();
            result.Dispose();
        }

        [Fact]
        public void Tensor_Multiplication()
        {
            var a = _gpuBackend.CreateTensor(new Shape(3), new float[] { 2, 3, 4 });
            var b = _gpuBackend.CreateTensor(new Shape(3), new float[] { 5, 6, 7 });
            var result = a * b;
            
            Assert.Equal(new float[] { 10, 18, 28 }, result.ToHost());
            
            a.Dispose();
            b.Dispose();
            result.Dispose();
        }

        [Fact]
        public void Tensor_MatMul()
        {
            var a = _gpuBackend.CreateTensor(new Shape(2, 3), new float[] { 1, 2, 3, 4, 5, 6 });
            var b = _gpuBackend.CreateTensor(new Shape(3, 2), new float[] { 7, 8, 9, 10, 11, 12 });
            var result = a.MatMul(b);
            
            Assert.Equal(new Shape(2, 2), result.Shape);
            var expected = new float[] { 58, 64, 139, 154 };
            Assert.Equal(expected, result.ToHost());
            
            a.Dispose();
            b.Dispose();
            result.Dispose();
        }

        [Fact]
        public void Tensor_Tanh()
        {
            var input = _gpuBackend.CreateTensor(new Shape(3), new float[] { -1, 0, 1 });
            var result = input.Tanh();
            var output = result.ToHost();
            
            Assert.Equal(-0.7616f, output[0], 1e-3f);
            Assert.Equal(0.0f, output[1], 1e-6f);
            Assert.Equal(0.7616f, output[2], 1e-3f);
            
            input.Dispose();
            result.Dispose();
        }

        [Fact]
        public void Tensor_ReLU()
        {
            var input = _gpuBackend.CreateTensor(new Shape(4), new float[] { -2, -1, 0, 1 });
            var result = input.ReLU();
            
            Assert.Equal(new float[] { 0, 0, 0, 1 }, result.ToHost());
            
            input.Dispose();
            result.Dispose();
        }

        [Fact]
        public void Backend_ConsistencyBetweenGpuAndCpu()
        {
            var data1 = new float[] { 1, 2, 3 };
            var data2 = new float[] { 4, 5, 6 };
            
            var gpuA = _gpuBackend.CreateTensor(new Shape(3), data1);
            var gpuB = _gpuBackend.CreateTensor(new Shape(3), data2);
            var gpuResult = gpuA + gpuB;
            
            var cpuA = _cpuBackend.CreateTensor(new Shape(3), data1);
            var cpuB = _cpuBackend.CreateTensor(new Shape(3), data2);
            var cpuResult = cpuA + cpuB;
            
            Assert.Equal(cpuResult.ToHost(), gpuResult.ToHost());
            
            gpuA.Dispose();
            gpuB.Dispose();
            gpuResult.Dispose();
            cpuA.Dispose();
            cpuB.Dispose();
            cpuResult.Dispose();
        }

        public void Dispose()
        {
            _gpuBackend?.Dispose();
            _cpuBackend?.Dispose();
        }
    }
}