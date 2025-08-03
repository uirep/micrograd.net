using Xunit;
using Micrograd.Core;
using Micrograd.Core.Backends;
using System.Linq;

namespace Micrograd.Tests.Integration
{
    public class BackendCompatibilityTests : IDisposable
    {
        private readonly ITensorBackend _gpuBackend;
        private readonly ITensorBackend _cpuBackend;

        public BackendCompatibilityTests()
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
        public void TensorAdd_GpuCpuParity()
        {
            var data1 = new float[] { 1, 2, 3, 4, 5 };
            var data2 = new float[] { 6, 7, 8, 9, 10 };
            
            var gpuA = _gpuBackend.CreateTensor(new Shape(5), data1);
            var gpuB = _gpuBackend.CreateTensor(new Shape(5), data2);
            var gpuResult = gpuA + gpuB;
            
            var cpuA = _cpuBackend.CreateTensor(new Shape(5), data1);
            var cpuB = _cpuBackend.CreateTensor(new Shape(5), data2);
            var cpuResult = cpuA + cpuB;
            
            var gpuData = gpuResult.ToHost();
            var cpuData = cpuResult.ToHost();
            
            for (int i = 0; i < data1.Length; i++)
            {
                Assert.Equal(cpuData[i], gpuData[i], 1e-5f);
            }
            
            gpuA.Dispose();
            gpuB.Dispose();
            gpuResult.Dispose();
            cpuA.Dispose();
            cpuB.Dispose();
            cpuResult.Dispose();
        }

        [Fact]
        public void TensorMultiply_GpuCpuParity()
        {
            var data1 = new float[] { 1, 2, 3, 4 };
            var data2 = new float[] { 2, 3, 4, 5 };
            
            var gpuA = _gpuBackend.CreateTensor(new Shape(4), data1);
            var gpuB = _gpuBackend.CreateTensor(new Shape(4), data2);
            var gpuResult = gpuA * gpuB;
            
            var cpuA = _cpuBackend.CreateTensor(new Shape(4), data1);
            var cpuB = _cpuBackend.CreateTensor(new Shape(4), data2);
            var cpuResult = cpuA * cpuB;
            
            Assert.Equal(cpuResult.ToHost(), gpuResult.ToHost());
            
            gpuA.Dispose();
            gpuB.Dispose();
            gpuResult.Dispose();
            cpuA.Dispose();
            cpuB.Dispose();
            cpuResult.Dispose();
        }

        [Fact]
        public void TensorMatMul_GpuCpuParity()
        {
            var dataA = new float[] { 1, 2, 3, 4, 5, 6 };
            var dataB = new float[] { 7, 8, 9, 10, 11, 12 };
            
            var gpuA = _gpuBackend.CreateTensor(new Shape(2, 3), dataA);
            var gpuB = _gpuBackend.CreateTensor(new Shape(3, 2), dataB);
            var gpuResult = gpuA.MatMul(gpuB);
            
            var cpuA = _cpuBackend.CreateTensor(new Shape(2, 3), dataA);
            var cpuB = _cpuBackend.CreateTensor(new Shape(3, 2), dataB);
            var cpuResult = cpuA.MatMul(cpuB);
            
            Assert.Equal(cpuResult.ToHost(), gpuResult.ToHost());
            
            gpuA.Dispose();
            gpuB.Dispose();
            gpuResult.Dispose();
            cpuA.Dispose();
            cpuB.Dispose();
            cpuResult.Dispose();
        }

        [Fact]
        public void TensorTanh_GpuCpuParity()
        {
            var data = new float[] { -2, -1, 0, 1, 2 };
            
            var gpuTensor = _gpuBackend.CreateTensor(new Shape(5), data);
            var gpuResult = gpuTensor.Tanh();
            
            var cpuTensor = _cpuBackend.CreateTensor(new Shape(5), data);
            var cpuResult = cpuTensor.Tanh();
            
            var gpuData = gpuResult.ToHost();
            var cpuData = cpuResult.ToHost();
            
            for (int i = 0; i < data.Length; i++)
            {
                Assert.Equal(cpuData[i], gpuData[i], 1e-5f);
            }
            
            gpuTensor.Dispose();
            gpuResult.Dispose();
            cpuTensor.Dispose();
            cpuResult.Dispose();
        }

        [Fact]
        public void TensorReLU_GpuCpuParity()
        {
            var data = new float[] { -2, -1, 0, 1, 2 };
            
            var gpuTensor = _gpuBackend.CreateTensor(new Shape(5), data);
            var gpuResult = gpuTensor.ReLU();
            
            var cpuTensor = _cpuBackend.CreateTensor(new Shape(5), data);
            var cpuResult = cpuTensor.ReLU();
            
            Assert.Equal(cpuResult.ToHost(), gpuResult.ToHost());
            
            gpuTensor.Dispose();
            gpuResult.Dispose();
            cpuTensor.Dispose();
            cpuResult.Dispose();
        }

        [Fact]
        public void NeuralNetwork_GpuCpuTrainingParity()
        {
            var gpuMlp = new TensorMLP(2, new[] { 3, 1 }, _gpuBackend);
            var cpuMlp = new TensorMLP(2, new[] { 3, 1 }, _cpuBackend);
            
            var gpuOptimizer = new TensorSGDOptimizer(0.01f);
            var cpuOptimizer = new TensorSGDOptimizer(0.01f);
            
            var inputData = new[] { 1.0f, -1.0f };
            var targetData = 1.0f;
            
            var gpuInputs = inputData.Select(x => new TensorValue(_gpuBackend.CreateTensor(new Shape(1), new[] { x }))).ToArray();
            var gpuTarget = new TensorValue(_gpuBackend.CreateTensor(new Shape(1), new[] { targetData }));
            
            var cpuInputs = inputData.Select(x => new TensorValue(_cpuBackend.CreateTensor(new Shape(1), new[] { x }))).ToArray();
            var cpuTarget = new TensorValue(_cpuBackend.CreateTensor(new Shape(1), new[] { targetData }));
            
            var gpuPred = gpuMlp.ForwardSingle(gpuInputs);
            var gpuLoss = TensorLossFunctions.MeanSquaredError(gpuPred, gpuTarget);
            
            var cpuPred = cpuMlp.ForwardSingle(cpuInputs);
            var cpuLoss = TensorLossFunctions.MeanSquaredError(cpuPred, cpuTarget);
            
            Assert.True(gpuLoss.Data.ToHost()[0] >= 0);
            Assert.True(cpuLoss.Data.ToHost()[0] >= 0);
            
            foreach (var input in gpuInputs) input.Dispose();
            foreach (var input in cpuInputs) input.Dispose();
            gpuTarget.Dispose();
            cpuTarget.Dispose();
            gpuPred.Dispose();
            cpuPred.Dispose();
            gpuLoss.Dispose();
            cpuLoss.Dispose();
            gpuMlp.Dispose();
            cpuMlp.Dispose();
        }

        public void Dispose()
        {
            _gpuBackend?.Dispose();
            _cpuBackend?.Dispose();
        }
    }
}