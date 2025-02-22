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


## Day 5: learning about pytorch nn-module, dataset and dataloader class
- **Learn about nn-module in detail**.
- **Implemented nn-module in existing neural network**.
- **Learn about Dataset and Dataloader classes along with its parameters**.
- **It helps to minimize the code complexity by calling abstact class from `torch.utils.data`**
- **Used to make the batches of dataset and optimise the network**
- **Saw significant increase in accuracy**.


## learning from youtube 
- source
- `CampusX`
- Stay tuned for more updates! 

