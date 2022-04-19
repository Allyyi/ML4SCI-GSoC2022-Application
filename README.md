# ML4SCI Task Analysis

[TOC]

## Interested Project

1. **Graph Neural Networks for End-to-End Particle Identification with the CMS Experiment** (*update)

2. **Vision Transformers for End-to-End Particle Reconstruction for the CMS Experiment**

## Repository Structure

1. The solution of common tasks are recorded in folder **Task1** and **Task2**.
2. The solution of specific task for Vision Transformer for particle reconstruction is recorded in folder **Task3**.
3. The solution and analysis of Graph Neural Network is recorded in folder **Task4**.
4. Analysis for experiment **Task1, Task2, and Task3** are witten in the following Experiment Analysis section. The analysis of GNN in **Task4** can be found in [Task4/GNN_Analysis.md](./Task4/GNN_Analysis.md).

## Experiment Analysis

### Electron/Photon Classification(Task 1&3)

#### Model Selection

The dataset for this task is a 32x32 matrix with two channels, hit energy and time. The two classes, Electron and Photon, have the same amount of data which means there is no need to resample the data to balance the dataset. We can use the hit energy channel and treat it as a 32x32x1 image. Then, perform end-to-end image binary classification. 

Inspired by the paper in link1, firstly, a simple MLP network is trained as a baseline in Pytorch. Then, a simple CNN and a shallow [ResNet](https://machinelearningknowledge.ai/keras-implementation-of-resnet-50-architecture-from-scratch/) are built with Keras. Lastly, a shallow Vision Transformer modified from [keras code example](https://keras.io/examples/vision/image_classification_with_vision_transformer/) is applied to this task. All of the models are trained from scratch. Their detailed architecture and performance are shown below:

**MLP**: 4 hidden layers with size 256, ReLU activation function and 0.5 dropout probability.

**CNN**: 4 convolution layers with kernel size 3x3 and ReLU activation, batch normalization, and max pooling.

**Shallow ResNet**: 2 residual blocks, one is convolution block, and the other is identity block. 

**Shallow ViT**: 1 layer transformer block.

|                   | **MLP**  | **CNN**  | **ResNet** | **ViT**  |
| ----------------- | -------- | -------- | ---------- | -------- |
| **Test Accuracy** | 0.726968 | 0.732450 | 0.726857   | 0.722861 |
| **Test AUC**      | 0.792841 | 0.800251 | 0.793122   | 0.794234 |

#### Problems and Solutions

- **Problem 1:** The loss and accuracy of training fluctuates and is stuck after the 10th epoch or so.

- - **Solution:** When this situation happens, it is most likely caused by an inappropriate learning rate. When the model is optimized to a specific state, it needs a lower learning rate for fine-tuning. However, if the learning rate is too low at the start, the convergence of loss will be slow. So, an **exponential learning rate decay** is applied in this case.

- **Problem 2**: When using Vision Transformer, the accuracy is still no better than random guess after training several batchs.

- - **Solution**: After ruling out the possibility of errors in the model network itself and calculating the loss function, the error may come from the data distribution. The problem is addressed by applying normalization to data before feeding to it the network.

#### Comparison between CNN and ViT

In the experiment, the shallow Vision Transformer shows promising results that it has achieved competing results compared with the convolution-based model. However, it takes more epochs to achieve this competing level and is more sensitive to differences in the learning rate, which could be explained by comparing their different mechanism:

- The convolution operation naturally possesses locality and translation equivariance when it comes to the image task. 

- The transformer block relies on the self-attention mechanism, which does not introduce the domain knowledge of image. 

Also, The CNN-based model has fewer trainable parameters than the vision transformer in this experiment. Thus, training with vision transformer requires more epochs.

#### Further Study

- The convolutional model and vision transformer have not reached their full potential. We can obtain better results by more refined hyperparameter tuning.
- In this experiment, the vision transformer model is a vanilla version. As vision transformer research progresses, we can also apply other more suitable methods to the small dataset, including compact vision transformer(CVT) and combing transformer with convolution.

### Quark/Gluon Classification(Task 2)

#### Model Selection

Due to the memory constraint with my computer, only 14000 data in the first given dataset is read, and its data type is set as float32. In this task, there are mainly two methods:

- Pure CNN architecture with only X_jets data as input.

  Since the X_jets data is sparse, when I am trying to use a more complicated model like shallow ResNet or stack more convolutional layers, the model suffers overfitting problems. The performance does not benefit from a more complicated structure. Thus, only one convolutional layer, batch norm, and max pooling are applied.

- CNN and linear layer architecture with mixed inputs, including X_jets, pt, and m0 features.

  For X_jets input, a similar convolutional layer is applied, but with a smaller channel(4), a larger kernel size (5x5), and a larger pooling size(4x4) due to the sparsity of data and in the hope of reducing training parameter to avoid overfitting issue best. For pt and m0 features, they are projected to a 4x1 vector and then followed by a Dense layer sized 8. Then, concatenate the output of X_jets, pt, and m0 and feed the vector to 2 linear layers for classification. 

The classification performance benefits from mixed inputs compared with only taking the X_jets. Their performance is shown below:

|                   | CNN      | Mixed Inputs |
| ----------------- | -------- | ------------ |
| **Test Accuracy** | 0.680000 | 0.702143     |
| **Test Auc**      | 0.731377 | 0.760988     |

#### Problems and Solutions

- **Problem 1**: In the mixed inputs model, I perform data standardize to scale the X_jets, pt, and m0 to [0,1]. However, the performance seems worse than a pure CNN model. 
  - **Solution:** By performing ablation experiment, I find that only performing standardization to pt and m0 and leaving X_jets as its original value is better than being not standardized at all or all standardized.
  
- **Problem 2**: Inappropriate learning rate.
- **Solution:** Using grid search method to find reasonable learning rate and modify decay step to suit the current case.

## Reflections

- Unlike the data in natural language processing and computer vision that I have been familiar with, raw physical data is susceptible to the ways it is being prepossessed. Even if the data is treated as an image, some preprocessing method such as random crop, random blur, and random rotation is not suitable for raw physical data of particles. Investigating the data value distribution is necessary.
- In current deep learning, it is common for a model to have a deep structure to obtain a good performance. However, when performing tasks in physics, a deep and complicated model may lead to overfitting and bad performance.  It is better to follow Occam's Razor principles rather than simply stacking existing methods when designing models.