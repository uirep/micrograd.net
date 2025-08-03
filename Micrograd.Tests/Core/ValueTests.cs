using Xunit;
using Micrograd.Core;
using System;
using System.Linq;

namespace Micrograd.Tests.Core
{
    public class ValueTests
    {
        [Fact]
        public void Value_BasicArithmetic()
        {
            var a = new Value(2.0);
            var b = new Value(3.0);
            var c = a + b;
            
            Assert.Equal(5.0, c.Data, 1e-10);
        }

        [Fact]
        public void Value_MultiplicationAndChaining()
        {
            var a = new Value(2.0);
            var b = new Value(3.0);
            var c = a * b;
            var d = c + a;
            
            Assert.Equal(6.0, c.Data, 1e-10);
            Assert.Equal(8.0, d.Data, 1e-10);
        }

        [Fact]
        public void Value_Backward_SimpleAddition()
        {
            var a = new Value(2.0);
            var b = new Value(3.0);
            var c = a + b;
            
            c.Backward();
            
            Assert.Equal(1.0, a.Grad, 1e-10);
            Assert.Equal(1.0, b.Grad, 1e-10);
        }

        [Fact]
        public void Value_Backward_SimpleMultiplication()
        {
            var a = new Value(2.0);
            var b = new Value(3.0);
            var c = a * b;
            
            c.Backward();
            
            Assert.Equal(3.0, a.Grad, 1e-10);
            Assert.Equal(2.0, b.Grad, 1e-10);
        }

        [Fact]
        public void Value_ComplexExpression()
        {
            var x = new Value(2.0);
            var y = new Value(3.0);
            var z = x * y + x * x;
            
            z.Backward();
            
            Assert.Equal(10.0, z.Data, 1e-10);
            Assert.Equal(7.0, x.Grad, 1e-10);
            Assert.Equal(2.0, y.Grad, 1e-10);
        }

        [Fact]
        public void Value_TanhActivation()
        {
            var x = new Value(0.0);
            var y = x.Tanh();
            
            y.Backward();
            
            Assert.Equal(0.0, y.Data, 1e-10);
            Assert.Equal(1.0, x.Grad, 1e-10);
        }

        [Fact]
        public void Value_ReLUActivation()
        {
            var x1 = new Value(-1.0);
            var x2 = new Value(1.0);
            var y1 = x1.ReLU();
            var y2 = x2.ReLU();
            
            y1.Backward();
            y2.Backward();
            
            Assert.Equal(0.0, y1.Data, 1e-10);
            Assert.Equal(1.0, y2.Data, 1e-10);
            Assert.Equal(0.0, x1.Grad, 1e-10);
            Assert.Equal(1.0, x2.Grad, 1e-10);
        }

        [Fact]
        public void Value_PowerOperation()
        {
            var x = new Value(3.0);
            var y = x.Pow(2.0);
            
            y.Backward();
            
            Assert.Equal(9.0, y.Data, 1e-10);
            Assert.Equal(6.0, x.Grad, 1e-10);
        }

        [Fact]
        public void Value_ExponentialFunction()
        {
            var x = new Value(0.0);
            var y = x.Exp();
            
            y.Backward();
            
            Assert.Equal(1.0, y.Data, 1e-10);
            Assert.Equal(1.0, x.Grad, 1e-10);
        }

        [Fact]
        public void Value_SigmoidFunction()
        {
            var x = new Value(0.0);
            var y = x.Sigmoid();
            
            y.Backward();
            
            Assert.Equal(0.5, y.Data, 1e-10);
            Assert.Equal(0.25, x.Grad, 1e-10);
        }

        [Fact]
        public void Value_Division()
        {
            var a = new Value(6.0);
            var b = new Value(2.0);
            var c = a / b;
            
            c.Backward();
            
            Assert.Equal(3.0, c.Data, 1e-10);
            Assert.Equal(0.5, a.Grad, 1e-10);
            Assert.Equal(-1.5, b.Grad, 1e-10);
        }

        [Fact]
        public void Value_Subtraction()
        {
            var a = new Value(5.0);
            var b = new Value(3.0);
            var c = a - b;
            
            c.Backward();
            
            Assert.Equal(2.0, c.Data, 1e-10);
            Assert.Equal(1.0, a.Grad, 1e-10);
            Assert.Equal(-1.0, b.Grad, 1e-10);
        }

        [Fact]
        public void Value_Negation()
        {
            var a = new Value(5.0);
            var b = -a;
            
            b.Backward();
            
            Assert.Equal(-5.0, b.Data, 1e-10);
            Assert.Equal(-1.0, a.Grad, 1e-10);
        }

        [Fact]
        public void Value_ScalarOperations()
        {
            var a = new Value(2.0);
            var b = a + 3.0;
            var c = 5.0 * a;
            
            b.Backward();
            a.ZeroGrad();
            c.Backward();
            
            Assert.Equal(5.0, b.Data, 1e-10);
            Assert.Equal(10.0, c.Data, 1e-10);
            Assert.Equal(5.0, a.Grad, 1e-10);
        }

        [Fact]
        public void Value_ZeroGrad()
        {
            var a = new Value(2.0);
            var b = new Value(3.0);
            var c = a * b;
            
            c.Backward();
            Assert.NotEqual(0.0, a.Grad);
            
            a.ZeroGrad();
            Assert.Equal(0.0, a.Grad, 1e-10);
            Assert.Equal(0.0, b.Grad, 1e-10);
        }

        [Fact]
        public void Value_ImplicitConversions()
        {
            Value a = 2.5;
            double b = a;
            
            Assert.Equal(2.5, a.Data, 1e-10);
            Assert.Equal(2.5, b, 1e-10);
        }

        [Fact]
        public void Value_Labels()
        {
            var a = new Value(2.0, label: "x");
            var b = new Value(3.0, label: "y");
            var c = a + b;
            c.Label = "sum";
            
            Assert.Equal("x", a.Label);
            Assert.Equal("y", b.Label);
            Assert.Equal("sum", c.Label);
        }
    }
}