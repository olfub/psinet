# $\Psi$net : Efficient Causal Modeling at Scale

Being a ubiquitous aspect of human cognition, causality has made its way into modern-day machine-learning research. Despite its importance in real-world applications, contemporary research still struggles with high-dimensional causal problems. Leveraging the efficiency of probabilistic circuits, which offer tractable computation of marginal probabilities, we introduce $\Psi$net, a probabilistic model designed for large-scale causal inference. $\Psi$net is a type of sum-product network where layering and the einsum operation allow for efficient parallelization. By incorporating interventional data into the learning process, the model can learn the effects of interventions and make predictions based on the specific interventional setting. Overall, $\Psi$net is a causal probabilistic circuit that efficiently answers causal queries in large-scale problems. We present evaluations conducted on both synthetic data and a substantial real-world dataset, demonstrating $\Psi$net's ability to capture causal relationships in high-dimensional settings.

## How to use the code

The synthetic experiments can be run using `./examples/synthetic/run_experiments.sh`

The experiments for CausalBench can be run using `./examples/run_causalbench_einet.sh`, `./examples/run_causalbench_iSPN.sh`, `./examples/run_causalbench_ncm.sh`

In order to replicate the experiments in the paper, be careful to set the values as describe in the experimental section in the appendix of the paper as batch size and runtime are not specified in the files to run the experiment but instead need to be changed in the code.