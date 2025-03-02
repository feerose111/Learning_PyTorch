# My Project
# Pytorch Learning Log

This repository documents my journey of learning **PyTorch** and its differences from **TensorFlow**.

## Day 1: Understanding PyTorch and Its Differences from TensorFlow
- Learned about **PyTorch** and how it differs from **TensorFlow**.
- PyTorch is more **dynamic and flexible**, using eager execution, whereas TensorFlow uses **static computation graphs** (though TF 2.x introduced eager execution by default).
- PyTorch is **preferred for research and experimentation**, while TensorFlow is more commonly used in **production and deployment**.
- PyTorch provides **better debugging** due to its Pythonic nature and support for standard debugging tools.

## Day 2: Basic Operations in PyTorch
- Explored basic **tensor operations** in PyTorch.
- Learned how to **create tensors**, perform **mathematical operations**, and manipulate **tensor shapes**.
- Experimented with **GPU acceleration** by moving tensors to CUDA (`tensor.to('cuda')`).
- Practiced operations like **addition, multiplication, reshaping, and indexing**.

## Day 3: Learning About Basics of Autograd
- Explored **automatic differentiation** using PyTorch's `autograd` module.
- Understood how PyTorch computes gradients automatically for tensor operations.
- Learned how to use `requires_grad=True` to track computations and compute gradients.
- Practiced using `backward()` and `grad` attributes to access gradient values.

## Day 4: Creating a Neural Network Using PyTorch from Scratch
- **Built a simple neural network**.
- **Trained it on a real-world dataset**.
- **Mimicked the PyTorch workflow**.
- **Used a lot of manual elements** to understand the internal workings.
- **Focused on learning rather than the final result**.
  
### Steps Followed:
1. **Loaded the dataset** 
2. **Performed basic preprocessing** 
3. **Training Process:**
   - a. Created the model
   - b. Forward pass
   - c. Loss calculation
   - d. Backpropagation
   - e. Parameter updates
4. **Model evaluation** (to be done in future sessions)


## Day 5: Learning about pytorch nn-module, dataset and dataloader class
- **Learn about nn-module in detail**.
- **Implemented nn-module in existing neural network**.
- **Learn about Dataset and Dataloader classes along with its parameters**.
- **It helps to minimize the code complexity by calling abstact class from `torch.utils.data`**
- **Used to make the batches of dataset and optimise the network**
- **Saw significant increase in accuracy**.
  
## Day 6: Building an ANN to predict label of FMNIST dataset from kaggle
- **Build an simple ANN from scratch using torch in Google colab GPU**
- **Ran the train and test for whole FMNIST data set of `70,000` images of `28*28` labeled dataset**
- **Achieved test accuracy at around `88%`**
- **Encountered problem of overfitting**
- **To overcome overfitting used techniques such as Batch Normalization and Dropout in the Neural Network**
- **Performed Regularization on model weight adding `weight-decay` parameter in the optimizer**
- **Then achieved training accuracy around `94%` and test accuracy around `89%` slightly reducing overfitting**

## Day 7: Learning to perform hyperparameter tuning on previously built model using optuna
- **Used Bayesian search for hyperparameter tuning**
- **Defined a function `objective` and passed `trail` class in that function**
- **Dynamic approach was used to pass the parameters in that function which allows optuna to try different set of parameters**
- **Trails were performed to find the best parameter example number of hidden layers, number of neurons per layer, dropout rate, learning rate, epochs, optimizer, etc.**

## Day 8: Learning how CNN is used in pytorch
- **In the model `nn.Conv2d ` is used to invoke a convolution function to extract the features**
- **Extracted features were sent to linear layer after flattening the feature map**
- **Saw that testing accuracy increased drastically after only using simple CNN architecture**

## Day 9: Learning transfer learning using pytorch
- **Chose `VGG16` as a pretrained model to run image classification**
- **Did all the necessary tranformation before sending the input image to `VGG16`**
- **Transformation included:**
  - resizing image to `256*256`
  - center cropping
  - converted to tensor
  - Normalize the image to given mean and std in pytorch documentation
- **Used the features extraction without changing(freezing the feature extraction)**
- **Made own classifier as we only need 10 classification class**
- **Trained the model**

## Day 10: Learning RNN architecture and implementing it using pytorch
- **Learn about the working mechanism of RNN**
- **Build a simple RNN architecture using pytorch**
- **Build a question answering system using RNN on a squential question-answer dataset**
- **Learn to preprocess a CSV question-answer dataset using pytorch before ending it to RNN**
- **Successfully implemented a simple QA system to answer user question that are only present in that particular dataset**

## learning from youtube 
- source:
-- `CampusX`
- Stay tuned for more updates! 

