using System;
using System.Collections.Generic;
using System.Linq;

namespace Micrograd.Core
{
    /// <summary>
    /// A value that supports automatic differentiation through the computational graph.
    /// This is the core building block of the micrograd library, implementing reverse-mode autodiff.
    /// </summary>
    public class Value
    {
        /// <summary>
        /// The actual numerical value stored in this node
        /// </summary>
        public double Data { get; set; }

        /// <summary>
        /// The gradient of this value with respect to the final output
        /// </summary>
        public double Grad { get; set; }

        /// <summary>
        /// The operation that created this value (for debugging/visualization)
        /// </summary>
        public string Op { get; private set; }

        /// <summary>
        /// The label for this value (for debugging/visualization)
        /// </summary>
        public string Label { get; set; }

        /// <summary>
        /// The children values that were used to compute this value
        /// </summary>
        private readonly HashSet<Value> _prev;

        /// <summary>
        /// The backward function that computes gradients for the children
        /// </summary>
        private Action _backward;

        /// <summary>
        /// Creates a new Value with the given data
        /// </summary>
        /// <param name="data">The numerical value</param>
        /// <param name="children">The child values that were used to compute this value</param>
        /// <param name="op">The operation that created this value</param>
        /// <param name="label">Optional label for debugging</param>
        public Value(double data, IEnumerable<Value>? children = null, string op = "", string label = "")
        {
            Data = data;
            Grad = 0.0;
            Op = op;
            Label = label;
            _prev = children?.ToHashSet() ?? new HashSet<Value>();
            _backward = () => { }; // Default backward does nothing
        }

        /// <summary>
        /// Addition operation
        /// </summary>
        public static Value operator +(Value a, Value b)
        {
            var result = new Value(a.Data + b.Data, new[] { a, b }, "+");
            
            result._backward = () =>
            {
                a.Grad += result.Grad;
                b.Grad += result.Grad;
            };
            
            return result;
        }

        /// <summary>
        /// Addition with a scalar
        /// </summary>
        public static Value operator +(Value a, double b)
        {
            return a + new Value(b);
        }

        /// <summary>
        /// Addition with a scalar (reverse order)
        /// </summary>
        public static Value operator +(double a, Value b)
        {
            return new Value(a) + b;
        }

        /// <summary>
        /// Multiplication operation
        /// </summary>
        public static Value operator *(Value a, Value b)
        {
            var result = new Value(a.Data * b.Data, new[] { a, b }, "*");
            
            result._backward = () =>
            {
                a.Grad += b.Data * result.Grad;
                b.Grad += a.Data * result.Grad;
            };
            
            return result;
        }

        /// <summary>
        /// Multiplication with a scalar
        /// </summary>
        public static Value operator *(Value a, double b)
        {
            return a * new Value(b);
        }

        /// <summary>
        /// Multiplication with a scalar (reverse order)
        /// </summary>
        public static Value operator *(double a, Value b)
        {
            return new Value(a) * b;
        }

        /// <summary>
        /// Negation operation
        /// </summary>
        public static Value operator -(Value a)
        {
            return a * new Value(-1);
        }

        /// <summary>
        /// Subtraction operation
        /// </summary>
        public static Value operator -(Value a, Value b)
        {
            return a + (-b);
        }

        /// <summary>
        /// Subtraction with a scalar
        /// </summary>
        public static Value operator -(Value a, double b)
        {
            return a - new Value(b);
        }

        /// <summary>
        /// Subtraction with a scalar (reverse order)
        /// </summary>
        public static Value operator -(double a, Value b)
        {
            return new Value(a) - b;
        }

        /// <summary>
        /// Division operation
        /// </summary>
        public static Value operator /(Value a, Value b)
        {
            return a * b.Pow(-1.0);
        }

        /// <summary>
        /// Division with a scalar
        /// </summary>
        public static Value operator /(Value a, double b)
        {
            return a / new Value(b);
        }

        /// <summary>
        /// Division with a scalar (reverse order)
        /// </summary>
        public static Value operator /(double a, Value b)
        {
            return new Value(a) / b;
        }

        /// <summary>
        /// Power operation
        /// </summary>
        public Value Pow(double other)
        {
            var result = new Value(Math.Pow(Data, other), new[] { this }, $"**{other}");
            
            result._backward = () =>
            {
                Grad += other * Math.Pow(Data, other - 1) * result.Grad;
            };
            
            return result;
        }

        /// <summary>
        /// Exponential function
        /// </summary>
        public Value Exp()
        {
            var result = new Value(Math.Exp(Data), new[] { this }, "exp");
            
            result._backward = () =>
            {
                Grad += result.Data * result.Grad;
            };
            
            return result;
        }

        /// <summary>
        /// Tanh activation function
        /// </summary>
        public Value Tanh()
        {
            var t = Math.Tanh(Data);
            var result = new Value(t, new[] { this }, "tanh");
            
            result._backward = () =>
            {
                Grad += (1 - t * t) * result.Grad;
            };
            
            return result;
        }

        /// <summary>
        /// ReLU activation function
        /// </summary>
        public Value ReLU()
        {
            var result = new Value(Data < 0 ? 0 : Data, new[] { this }, "ReLU");
            
            result._backward = () =>
            {
                Grad += (result.Data > 0 ? 1 : 0) * result.Grad;
            };
            
            return result;
        }

        /// <summary>
        /// Sigmoid activation function
        /// </summary>
        public Value Sigmoid()
        {
            var expNegX = (-this).Exp();
            return 1.0 / (1.0 + expNegX);
        }

        /// <summary>
        /// Performs backpropagation from this value through the computational graph
        /// </summary>
        public void Backward()
        {
            // Topological sort to get the correct order for backpropagation
            var topo = new List<Value>();
            var visited = new HashSet<Value>();
            
            void BuildTopo(Value v)
            {
                if (!visited.Contains(v))
                {
                    visited.Add(v);
                    foreach (var child in v._prev)
                    {
                        BuildTopo(child);
                    }
                    topo.Add(v);
                }
            }
            
            BuildTopo(this);
            
            // Initialize gradient of output to 1
            Grad = 1.0;
            
            // Go through all values in reverse topological order and apply chain rule
            for (int i = topo.Count - 1; i >= 0; i--)
            {
                topo[i]._backward();
            }
        }

        /// <summary>
        /// Zeros out all gradients in the computational graph
        /// </summary>
        public void ZeroGrad()
        {
            var visited = new HashSet<Value>();
            
            void ZeroGradRecursive(Value v)
            {
                if (!visited.Contains(v))
                {
                    visited.Add(v);
                    v.Grad = 0.0;
                    foreach (var child in v._prev)
                    {
                        ZeroGradRecursive(child);
                    }
                }
            }
            
            ZeroGradRecursive(this);
        }

        /// <summary>
        /// String representation of the value
        /// </summary>
        public override string ToString()
        {
            var label = !string.IsNullOrEmpty(Label) ? $"{Label}:" : "";
            return $"Value({label}data={Data:F4}, grad={Grad:F4})";
        }

        /// <summary>
        /// Implicit conversion from double to Value
        /// </summary>
        public static implicit operator Value(double value)
        {
            return new Value(value);
        }

        /// <summary>
        /// Implicit conversion from Value to double (gets the data)
        /// </summary>
        public static implicit operator double(Value value)
        {
            return value.Data;
        }
    }
} 