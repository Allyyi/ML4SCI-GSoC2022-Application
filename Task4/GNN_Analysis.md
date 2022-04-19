# GNN Experiment Analysis

### Jet Representation

The effectiveness of ML techniques on jet physics relies heavily on how the jet is represented. In Task 2, the jets are represented as image. Calorimeters measure the energy of a jet on fine-grained spatial cells. The energy deposition is treated as pixel intensity, which natrually creates an image for jet. However, this method leaves many blanks in image that contributes little useful information. A natural thought is to represent data in a graph form. 

**Node**: The pixel along X_jets channel. Thus, a node has three features. Each of them represents the ECAL, HCAL, and Track. To address the problem of empty blank in jets image, I only count non-zero pixels as nodes.

**Edge**: Because the physical meaning of edges in high-energy particles is not clear as that in Social networks or Molecular Biology, there are many ways in defining edges and connectivity. One common practice is to determine the graph as a fully-connected graph. However, after investigating data, the number of nodes in each graph ranges from approximately (400, 900) so it will add too much computational complexity and memory cost. Also, another way treats pixels as points cloud and does not consider edges. In the balance between both, I measured the Euclidean distance between nodes and connected the nodes with their top-k neighbors undirectely. In my experiments, the k is set to be 10.

### Model Selection

Due to the memory limitaiton of my CPU and GPU, I was able to read 10,000 quark/gluon data from run0 dataset. The data used for training is well-balanced so there no need to resample the data and both accuracy and AUC score can reflect the model perforamce well. Here we use the accuracy as evaluation metrics.

#### Architecture

The model architectures are shown in the following figure. They adpot the same training strategy,  including Cross Entropy Loss function, Adam Optimizer function, and similar classification head.

<img src="https://imgtu.com/i/LBKI6s" alt="image-20220419155408772" style="zoom:40%;" />

#### Comparison

|                         | **Model 1** | **Model 2** |
| ----------------------- | ----------- | ----------- |
| **Validation Accuracy** | 73.85%      | 71.80%      |
| **Test Accuracy**       | 72.89%      | 71.59%      |

These DNN models outperform CNN models in Task 2 in terms of accuracy. And model I is slightly better than model II. Due to the considerable difference in architecture, we can't pinpoint which graph block plays the key part in improvement, but we can investigate it through ablation studys in future work. A more detailed network architecture design for E2E tau identification is discussed in proposal.

