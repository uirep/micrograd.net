using Xunit;
using Micrograd.Core;
using System;

namespace Micrograd.Tests
{
    public class ValueTests
    {
        private const double Tolerance = 1e-6;

        [Fact]
        public void Value_Constructor_SetsDataCorrectly()
        {
            var value = new Value(3.14);
            Assert.Equal(3.14, value.Data, Tolerance);
            Assert.Equal(0.0, value.Grad, Tolerance);
        }

        [Fact]
        public void Value_Addition_WorksCorrectly()
        {
            var a = new Value(2.0);
            var b = new Value(3.0);
            var c = a + b;
            
            Assert.Equal(5.0, c.Data, Tolerance);
        }

        [Fact]
        public void Value_Addition_BackwardPassCorrect()
        {
            var a = new Value(2.0);
            var b = new Value(3.0);
            var c = a + b;
            
            c.Backward();
            
            Assert.Equal(1.0, a.Grad, Tolerance);
            Assert.Equal(1.0, b.Grad, Tolerance);
        }

        [Fact]
        public void Value_Multiplication_WorksCorrectly()
        {
            var a = new Value(2.0);
            var b = new Value(3.0);
            var c = a * b;
            
            Assert.Equal(6.0, c.Data, Tolerance);
        }

        [Fact]
        public void Value_Multiplication_BackwardPassCorrect()
        {
            var a = new Value(2.0);
            var b = new Value(3.0);
            var c = a * b;
            
            c.Backward();
            
            Assert.Equal(3.0, a.Grad, Tolerance); // dc/da = b.Data = 3.0
            Assert.Equal(2.0, b.Grad, Tolerance); // dc/db = a.Data = 2.0
        }

        [Fact]
        public void Value_Subtraction_WorksCorrectly()
        {
            var a = new Value(5.0);
            var b = new Value(3.0);
            var c = a - b;
            
            Assert.Equal(2.0, c.Data, Tolerance);
        }

        [Fact]
        public void Value_Division_WorksCorrectly()
        {
            var a = new Value(6.0);
            var b = new Value(2.0);
            var c = a / b;
            
            Assert.Equal(3.0, c.Data, Tolerance);
        }

        [Fact]
        public void Value_Power_WorksCorrectly()
        {
            var a = new Value(2.0);
            var c = a.Pow(3);
            
            Assert.Equal(8.0, c.Data, Tolerance);
        }

        [Fact]
        public void Value_Power_BackwardPassCorrect()
        {
            var a = new Value(2.0);
            var c = a.Pow(3);
            
            c.Backward();
            
            // dc/da = 3 * a^2 = 3 * 4 = 12
            Assert.Equal(12.0, a.Grad, Tolerance);
        }

        [Fact]
        public void Value_Exp_WorksCorrectly()
        {
            var a = new Value(1.0);
            var c = a.Exp();
            
            Assert.Equal(Math.E, c.Data, Tolerance);
        }

        [Fact]
        public void Value_Tanh_WorksCorrectly()
        {
            var a = new Value(0.0);
            var c = a.Tanh();
            
            Assert.Equal(0.0, c.Data, Tolerance);
            
            var b = new Value(1.0);
            var d = b.Tanh();
            
            Assert.Equal(Math.Tanh(1.0), d.Data, Tolerance);
        }

        [Fact]
        public void Value_ReLU_WorksCorrectly()
        {
            // Positive input
            var a = new Value(2.0);
            var c = a.ReLU();
            Assert.Equal(2.0, c.Data, Tolerance);
            
            // Negative input
            var b = new Value(-1.0);
            var d = b.ReLU();
            Assert.Equal(0.0, d.Data, Tolerance);
        }

        [Fact]
        public void Value_Sigmoid_WorksCorrectly()
        {
            var a = new Value(0.0);
            var c = a.Sigmoid();
            
            Assert.Equal(0.5, c.Data, Tolerance);
        }

        [Fact]
        public void Value_ComplexExpression_BackwardPassCorrect()
        {
            // Test: f(x,y) = x*y + sin(x) using polynomial approximation
            // Let's use f(x,y) = x*y + x^2 for simplicity
            var x = new Value(2.0);
            var y = new Value(3.0);
            
            var xy = x * y;
            var x2 = x * x;
            var f = xy + x2;
            
            f.Backward();
            
            // df/dx = y + 2*x = 3 + 2*2 = 7
            // df/dy = x = 2
            Assert.Equal(7.0, x.Grad, Tolerance);
            Assert.Equal(2.0, y.Grad, Tolerance);
        }

        [Fact]
        public void Value_ChainRule_WorksCorrectly()
        {
            // Test chain rule: f = (x + y) * (x - y) = x^2 - y^2
            var x = new Value(3.0);
            var y = new Value(2.0);
            
            var sum = x + y;
            var diff = x - y;
            var f = sum * diff;
            
            f.Backward();
            
            // f = x^2 - y^2, so df/dx = 2*x = 6, df/dy = -2*y = -4
            Assert.Equal(6.0, x.Grad, Tolerance);
            Assert.Equal(-4.0, y.Grad, Tolerance);
        }

        [Fact]
        public void Value_ZeroGrad_ResetsGradients()
        {
            var x = new Value(2.0);
            var y = new Value(3.0);
            var f = x * y;
            
            f.Backward();
            Assert.NotEqual(0.0, x.Grad);
            Assert.NotEqual(0.0, y.Grad);
            
            f.ZeroGrad();
            Assert.Equal(0.0, x.Grad, Tolerance);
            Assert.Equal(0.0, y.Grad, Tolerance);
        }

        [Fact]
        public void Value_ImplicitConversion_WorksCorrectly()
        {
            Value v = 3.14; // double to Value
            Assert.Equal(3.14, v.Data, Tolerance);
            
            double d = v; // Value to double
            Assert.Equal(3.14, d, Tolerance);
        }

        [Fact]
        public void Value_ScalarOperations_WorkCorrectly()
        {
            var a = new Value(2.0);
            
            var b = a + 3.0;
            Assert.Equal(5.0, b.Data, Tolerance);
            
            var c = 3.0 + a;
            Assert.Equal(5.0, c.Data, Tolerance);
            
            var d = a * 4.0;
            Assert.Equal(8.0, d.Data, Tolerance);
            
            var e = 4.0 * a;
            Assert.Equal(8.0, e.Data, Tolerance);
        }

        [Fact]
        public void Value_ToString_FormatsCorrectly()
        {
            var value = new Value(3.1415, label: "pi");
            value.Grad = 1.2345;
            
            var str = value.ToString();
            Assert.Contains("pi:", str);
            Assert.Contains("3.1415", str);
            Assert.Contains("1.2345", str);
        }
    }
} 