using System;
using System.Collections.Generic;
using System.Linq;

namespace Micrograd.Core
{

    public class Value
    {
        public double Data { get; set; }
        public double Grad { get; set; }
        public string Op { get; private set; }
        public string Label { get; set; }

        private readonly HashSet<Value> _prev;
        private Action _backward;

        public Value(double data, IEnumerable<Value>? children = null, string op = "", string label = "")
        {
            Data = data;
            Grad = 0.0;
            Op = op;
            Label = label;
            _prev = children?.ToHashSet() ?? new HashSet<Value>();
            _backward = () => { };
        }

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

        public static Value operator +(Value a, double b)
        {
            return a + new Value(b);
        }

        public static Value operator +(double a, Value b)
        {
            return new Value(a) + b;
        }

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

        public static Value operator *(Value a, double b)
        {
            return a * new Value(b);
        }

        public static Value operator *(double a, Value b)
        {
            return new Value(a) * b;
        }

        public static Value operator -(Value a)
        {
            return a * new Value(-1);
        }

        public static Value operator -(Value a, Value b)
        {
            return a + (-b);
        }

        public static Value operator -(Value a, double b)
        {
            return a - new Value(b);
        }

        public static Value operator -(double a, Value b)
        {
            return new Value(a) - b;
        }

        public static Value operator /(Value a, Value b)
        {
            return a * b.Pow(-1.0);
        }

        public static Value operator /(Value a, double b)
        {
            return a / new Value(b);
        }

        public static Value operator /(double a, Value b)
        {
            return new Value(a) / b;
        }

        public Value Pow(double other)
        {
            var result = new Value(Math.Pow(Data, other), new[] { this }, $"**{other}");
            
            result._backward = () =>
            {
                Grad += other * Math.Pow(Data, other - 1) * result.Grad;
            };
            
            return result;
        }

        public Value Exp()
        {
            var result = new Value(Math.Exp(Data), new[] { this }, "exp");
            
            result._backward = () =>
            {
                Grad += result.Data * result.Grad;
            };
            
            return result;
        }

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

        public Value ReLU()
        {
            var result = new Value(Data < 0 ? 0 : Data, new[] { this }, "ReLU");
            
            result._backward = () =>
            {
                Grad += (result.Data > 0 ? 1 : 0) * result.Grad;
            };
            
            return result;
        }

        public Value Sigmoid()
        {
            var expNegX = (-this).Exp();
            return 1.0 / (1.0 + expNegX);
        }

        public void Backward()
        {
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
            
            Grad = 1.0;
            
            for (int i = topo.Count - 1; i >= 0; i--)
            {
                topo[i]._backward();
            }
        }

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

        public override string ToString()
        {
            var label = !string.IsNullOrEmpty(Label) ? $"{Label}:" : "";
            return $"Value({label}data={Data:F4}, grad={Grad:F4})";
        }

        public static implicit operator Value(double value)
        {
            return new Value(value);
        }

        public static implicit operator double(Value value)
        {
            return value.Data;
        }
    }
}