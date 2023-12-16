# COMBHelper: A Neural Approach to Reduce Search Space for Combinatorial Optimization Algorithms over Graphs

This is a PyTorch implementation of "COMBHelper: A Neural Approach to Reduce Search Space for Combinatorial Optimization Algorithms over Graphs".

## Requirements
See `requriements.txt` for required python libraries .

Test on a server with a dual-core Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz, 256 GB memory, NVIDIA RTX 3090 GPU, and Ubuntu 18.04.6 LTS operating system.

## Experiments

### Quick Start
Best teacher and student models on synthetic and real-world datasets are saved in "./syntheitc/best_model" and "./realworld/best_model" respectively. And the corresponding predictions are saved in "./realworld/preds" and "./syntheitc/preds".

You can just run the following command in the terminal to obtain the results in the paper:
```
python combhelper.py MVC GD OTC
```
where "MVC" means the Minimum Vertex Cover (MVC) problem, "GD" means the greedy algorithm and "OTC" is the dataset name.

### Train
If you want to train, you can follow the instructions below.
We will take the MVC problem on real-world datasets as an example. Run the following command in the terminal to train the teacher and student models:
```
cd realworld
python teacher_mvc.py
python student_mvc.py
```
The models will be saved in "./realworld/model".


### Prediction

Run the following command to predict the node labels and prune the search space:
```
python eval_mvc.py
```
Good nodes will be saved in "./realworld/preds".

### Solution Generation

Then just run the command in the 'Quick Start' section to generate solutions.

### Notice
1. You can modify all the commands above to get the results of other cases.
2. Note to comment the codes for MVC and uncomment the codes for MIS in `datasets.py` when CO problem changes (default code is for MVC). 
3. For the CPLEX library, you need to install the academic edition, otherwise LP will fail on large graphs.

## Questions
If you have any question, please feel free to contact me at t67729037@163.com