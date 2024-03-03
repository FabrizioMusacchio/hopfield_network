# Understanding Hebbian learning in Hopfield networks

Hopfield networks, a form of recurrent neural network (RNN), serve as a fundamental model for understanding associative memory and pattern recognition in computational neuroscience. Central to the operation of Hopfield networks is the Hebbian learning rule, an idea encapsulated by the maxim "neurons that fire together, wire together". In this [tutorial](https://www.fabriziomusacchio.com/blog/2024-03-03-hebbian_learning_and_hopfiled_networks/), we explore the mathematical underpinnings of Hebbian learning within Hopfield networks, emphasizing its role in pattern recognition.

For reproducibility:

```powershell
conda create -n hopfield python=3.10
conda activate hopfield
conda install -y mamba
mamba install -y numpy matplotlib
```