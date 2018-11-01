using Encog.Engine.Network.Activation;
using Encog.ML.Data.Basic;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace xor_demo
{
    class Program
    {
        static void Main(string[] args)
        {
            double[][] XOR_Input =
            {
                new[] {0.0, 0.0},
                new[] {1.0,0.0},
                new[] {0.0,1.0},
                new[] {1.0,1.0}
            };

            double[][] XOR_Ideal =
            {
                new[] {0.0 },
                new[] {1.0 },
                new[] {1.0},
                new[] {0.0}
            };

            var trainingSet = new BasicMLDataSet(XOR_Input, XOR_Ideal);

            BasicNetwork network = CreateNetwork();

            var train = new ResilientPropagation(network, trainingSet);

            int epoch = 1;
            do
            {
                train.Iteration();
                Console.WriteLine("Iteration # {0}, Error: {1}", epoch, train.Error);
                epoch++;

            } while (train.Error > 0.001);

            foreach (var item in trainingSet)
            {
                var output = network.Compute(item.Input);
                Console.WriteLine("Input : {0}, {1} ;; Ideal : {2} ;; Actual : {3}", item.Input[0], item.Input[1], item.Ideal[0], output[0]);
            }

            Console.Write("Press any key to exit..");
            Console.ReadLine();

        }

        private static BasicNetwork CreateNetwork()
        {
            var network = new BasicNetwork();
            network.AddLayer(new BasicLayer(null, true, 2));                            // Input Layer, 1 bias neuron
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 2));         // Hidden Layer, 1 bias neuron
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), false,1));         //Output Layer
            network.Structure.FinalizeStructure();
            network.Reset();
            return network;

        }
    }
}
