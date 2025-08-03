using Xunit;
using Micrograd.Core;
using Micrograd.Core.Backends;

namespace Micrograd.Tests.Tensors
{
    public class TensorAutogradTests : IDisposable
    {
        private readonly ITensorBackend _backend;

        public TensorAutogradTests()
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
        public void TensorValue_Autograd_Addition()
        {
            var a = new TensorValue(_backend.CreateTensor(new Shape(1), new float[] { 2 }));
            var b = new TensorValue(_backend.CreateTensor(new Shape(1), new float[] { 3 }));
            var c = a + b;
            
            c.Backward();
            
            Assert.Equal(5.0f, c.Data.ToHost()[0]);
            Assert.Equal(1.0f, a.Grad.ToHost()[0]);
            Assert.Equal(1.0f, b.Grad.ToHost()[0]);
            
            a.Dispose();
            b.Dispose();
            c.Dispose();
        }

        [Fact]
        public void TensorValue_Autograd_Multiplication()
        {
            var a = new TensorValue(_backend.CreateTensor(new Shape(1), new float[] { 2 }));
            var b = new TensorValue(_backend.CreateTensor(new Shape(1), new float[] { 3 }));
            var c = a * b;
            
            c.Backward();
            
            Assert.Equal(6.0f, c.Data.ToHost()[0]);
            Assert.Equal(3.0f, a.Grad.ToHost()[0]);
            Assert.Equal(2.0f, b.Grad.ToHost()[0]);
            
            a.Dispose();
            b.Dispose();
            c.Dispose();
        }

        [Fact]
        public void TensorValue_Autograd_ComplexGraph()
        {
            var x = new TensorValue(_backend.CreateTensor(new Shape(1), new float[] { 2 }));
            var y = new TensorValue(_backend.CreateTensor(new Shape(1), new float[] { 3 }));
            
            var z = (x * y) + (x * x);
            
            z.Backward();
            
            Assert.Equal(10.0f, z.Data.ToHost()[0]);
            Assert.Equal(7.0f, x.Grad.ToHost()[0]);
            Assert.Equal(2.0f, y.Grad.ToHost()[0]);
            
            x.Dispose();
            y.Dispose();
            z.Dispose();
        }

        [Fact]
        public void TensorValue_Tanh_Autograd()
        {
            var x = new TensorValue(_backend.CreateTensor(new Shape(1), new float[] { 0 }));
            var y = x.Tanh();
            
            y.Backward();
            
            Assert.Equal(0.0f, y.Data.ToHost()[0], 1e-6f);
            Assert.Equal(1.0f, x.Grad.ToHost()[0], 1e-6f);
            
            x.Dispose();
            y.Dispose();
        }

        [Fact]
        public void TensorValue_ReLU_Autograd()
        {
            var x1 = new TensorValue(_backend.CreateTensor(new Shape(1), new float[] { -1 }));
            var x2 = new TensorValue(_backend.CreateTensor(new Shape(1), new float[] { 1 }));
            
            var y1 = x1.ReLU();
            var y2 = x2.ReLU();
            
            y1.Backward();
            y2.Backward();
            
            Assert.Equal(0.0f, y1.Data.ToHost()[0]);
            Assert.Equal(1.0f, y2.Data.ToHost()[0]);
            Assert.Equal(0.0f, x1.Grad.ToHost()[0]);
            Assert.Equal(1.0f, x2.Grad.ToHost()[0]);
            
            x1.Dispose();
            x2.Dispose();
            y1.Dispose();
            y2.Dispose();
        }

        [Fact]
        public void TensorValue_MatMul_Autograd()
        {
            var a = new TensorValue(_backend.CreateTensor(new Shape(2, 2), new float[] { 1, 2, 3, 4 }));
            var b = new TensorValue(_backend.CreateTensor(new Shape(2, 2), new float[] { 5, 6, 7, 8 }));
            
            var c = a.MatMul(b);
            c.Backward();
            
            Assert.Equal(new Shape(2, 2), c.Data.Shape);
            Assert.NotNull(a.Grad);
            Assert.NotNull(b.Grad);
            
            a.Dispose();
            b.Dispose();
            c.Dispose();
        }

        [Fact]
        public void TensorValue_ZeroGrad()
        {
            var x = new TensorValue(_backend.CreateTensor(new Shape(1), new float[] { 2 }));
            var y = x * x;
            
            y.Backward();
            Assert.NotEqual(0.0f, x.Grad.ToHost()[0]);
            
            x.ZeroGrad();
            Assert.Equal(0.0f, x.Grad.ToHost()[0]);
            
            x.Dispose();
            y.Dispose();
        }

        [Fact]
        public void TensorValue_ToString()
        {
            var x = new TensorValue(_backend.CreateTensor(new Shape(2, 3), new float[6]), label: "test");
            var str = x.ToString();
            
            Assert.Contains("TensorValue", str);
            Assert.Contains("test", str);
            
            x.Dispose();
        }

        public void Dispose()
        {
            _backend?.Dispose();
        }
    }
}