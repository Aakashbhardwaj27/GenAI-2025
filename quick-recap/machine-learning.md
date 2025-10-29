# 🧠 Machine Learning & Deep Learning Revision Sheet

This is a **crashcourse revision sheet** summarizing the core concepts, key algorithms, architectures, and formulas for Machine Learning (ML) and Deep Learning (DL). It's designed for quick reference and last-minute preparation.

## ⚙️ 1. Core Concepts

| Concept | Key Idea | Formula / Notes | Learning Type |
| :--- | :--- | :--- | :--- |
| **Learning** | Finding a function $f(x) \approx y$ from data | Loss minimization: $\min_\theta L(y, f_\theta(x))$ | Universal |
| **Supervised Learning** | Learn from **labeled** data | Regression, Classification | Data $\rightarrow$ Output |
| **Unsupervised Learning** | Learn **structure** in unlabeled data | Clustering, PCA, Autoencoders | Data $\rightarrow$ Insight |
| **Reinforcement Learning (RL)** | Learn by interacting with environment | Reward maximization $\max_\pi \mathbb{E}[R_t]$ | Action $\rightarrow$ Reward |
| **Bias–Variance Tradeoff** | Bias = **underfit**, Variance = **overfit** | Total error = $\text{Bias}^2 + \text{Variance} + \text{Noise}$ | Error Analysis |

---

## 📈 2. Key ML Algorithms

| Algorithm | Core Idea | Notes / Key Equation |
| :--- | :--- | :--- |
| **Linear Regression** | Fit line minimizing MSE | $y = w^T x + b$; $L = \frac{1}{n}\sum(y - \hat{y})^2$ |
| **Logistic Regression** | Probabilistic classification | $\hat{y} = \sigma(w^T x + b)$, uses **cross-entropy loss** |
| **Naive Bayes** | Assume feature independence | $P(\text{class} \mid \text{features}) \propto P(\text{features} \mid \text{class}) P(\text{class})$ |
| **Decision Tree** | Split to maximize information gain | Uses **Entropy**: $H = -\sum p_i \log p_i$ |
| **Random Forest** | Ensemble of trees (**voting**) | Reduces **variance** (overfitting) |
| **Support Vector Machine (SVM)** | Maximize **margin** between classes | $\min \|w\|^2$ subject to margin constraints |
| **K-Nearest Neighbors (KNN)** | Classify by neighbor majority | **No training**, high inference cost |
| **K-Means** | Group by distance to centroid | $\min \sum_i \sum_{x \in S_i} \|x - \mu_i\|^2$ |
| **PCA** | Reduce dimensions, preserve variance | Eigenvectors of **covariance matrix** $X^T X$ |
| **Gradient Boosting / XGBoost**| Sequentially fix residuals | Strong learner from weak learners (e.g., small trees) |

---

## 🧮 3. Optimization

| Concept | Formula | Intuition |
| :--- | :--- | :--- |
| **Gradient Descent** | $\theta := \theta - \eta \nabla_\theta L(\theta)$ | Move **opposite to the slope** to find the minimum of the loss function. |
| **Stochastic GD (SGD)** | Update per sample/small batch | Faster, but noisier updates; helps avoid local minima. |
| **Momentum** | $v_t = \beta v_{t-1} + (1-\beta)\nabla L$ | Smooths updates, helps accelerate descent in relevant directions. |
| **Adam** | Combines Momentum + RMSProp | **Adaptive learning rate** for each parameter. |
| **Learning Rate Schedules**| Decay over time (e.g., step, cosine) | Avoids oscillations and helps converge to a better minimum. |

---

## 🧠 4. Deep Learning Fundamentals

| Concept | Description | Formula / Intuition |
| :--- | :--- | :--- |
| **Neuron** | Weighted sum + activation | $a = \sigma(Wx + b)$ |
| **Activation Functions** | Add non-linearity | Sigmoid, **ReLU** ($\max(0, x)$), $\tanh$, GELU |
| **Forward Pass** | Compute output | Layer-by-layer propagation from input to output. |
| **Backward Pass (Backprop)** | Chain rule to compute gradients | $\frac{∂L}{∂W} = \frac{∂L}{∂a}\frac{∂a}{∂z}\frac{∂z}{∂W}$ (Core training engine) |
| **Loss Functions** | Measure error | **MSE** (regression), **Cross-Entropy** (classification) |
| **Regularization** | Reduce overfitting | **L1** (sparsity), **L2** (weight decay), **Dropout** |

---

## 🧱 5. Neural Network Architectures

### 🧩 Feedforward Neural Network (ANN)
* **Description:** Stack of linear $\rightarrow$ nonlinear $\rightarrow$ output layers.
* **Use:** Universal function approximator. Prone to overfitting on small data.

### 🧩 Convolutional Neural Network (CNN)
* **Key Idea:** Local receptive fields, **weight sharing**.
* **Benefit:** Efficient & translation invariant.
* **Layers:** Conv $\rightarrow$ Pool $\rightarrow$ FC $\rightarrow$ Softmax.
* **Use:** Vision tasks, 2D signal processing. 

### 🧩 Recurrent Neural Network (RNN)
* **Key Idea:** Maintains **memory** over sequences.
* **Formula:** $h_t = f(Wx_t + U h_{t-1} + b)$
* **Challenge:** Suffers from **vanishing gradient** $\rightarrow$ fixed by **LSTM/GRU**.
* **Use:** Time series, simple text processing.

### 🧩 Transformers
* **Key Idea:** **Self-attention** replaces recurrence.
* **Attention Formula:** $\text{Attention}(Q,K,V) = \text{softmax}(QK^T / \sqrt{d_k})V$
* **Benefit:** Enables **parallelization**, foundation models (LLMs, Vision Transformers).
* **Use:** State-of-the-art for sequence data (text, long-range dependencies).

---

## 🔍 6. Regularization and Generalization

| Technique | Purpose | Idea |
| :--- | :--- | :--- |
| **Dropout** | Prevent co-adaptation | Randomly deactivate neurons during training. |
| **Batch Normalization** | Normalize activations | Standardizes inputs to a layer to stabilize/speed up training. |
| **Weight Decay (L2)** | Penalize large weights | Adds $\frac{\lambda}{2} \|w\|^2$ to the loss. |
| **Early Stopping** | Stop before overfitting | Monitor validation loss; halt training when loss starts increasing. |
| **Data Augmentation** | Artificially expand data | Improves robustness and generalization. |

---

## 🧩 7. Advanced Topics

| Topic | Key Idea | Quick Summary |
| :--- | :--- | :--- |
| **Autoencoders** | Learn compressed representation | Encoder–decoder structure, minimizes reconstruction loss. |
| **Generative Adversarial Networks (GANs)** | Generator vs Discriminator | Minimax game: $\min_G \max_D V(D,G)$. Generates highly realistic data. |
| **Diffusion Models** | Learn to denoise | Gradual noise removal to generate high-quality data. |
| **Transfer Learning** | Use pre-trained models | Fine-tune a large model on a new, smaller task. |
| **Self-Supervised Learning (SSL)** | Predict parts of data from other parts | Foundation for modern LLMs and Vision models (pre-training). |

---

## 📊 8. Evaluation Metrics

| Task | Primary Metrics | Description |
| :--- | :--- | :--- |
| **Classification** | Accuracy, Precision, Recall, F1, **ROC-AUC** | Measures balance between false positives and false negatives. |
| **Regression** | **MSE**, MAE, $R^2$ | Quantifies the magnitude of the prediction error. |
| **Clustering** | Silhouette Score, Adjusted Rand Index | Compares clustering quality/compactness. |
| **Generative** | FID, Inception Score | Evaluate the realism and quality of generated content. |

---

## ⚡ 9. Key Formulas Recap

| Concept | Formula |
| :--- | :--- |
| **Mean Squared Error (MSE)** | $\frac{1}{n}\sum(y - \hat{y})^2$ |
| **Cross-Entropy Loss** | $-\sum y \log(\hat{y})$ |
| **Softmax** | $\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$ (Converts logits to probabilities) |
| **Sigmoid** | $\sigma(x) = \frac{1}{1 + e^{-x}}$ |
| **ReLU** | $\max(0, x)$ |
| **Gradient Descent** | $\theta_{t+1} = \theta_t - \eta \nabla_\theta L$ |

---

## 🧠 10. Mental Models & Intuitions

* **Gradient Descent:** Like **rolling downhill** to the minimum error.
* **Regularization:** **Simpler models** generalize better (Occam's razor).
* **Ensemble Methods (e.g., Random Forest):** The **"Wisdom of crowds"** in prediction.
* **CNNs:** Learn visual **hierarchies** (edges $\rightarrow$ shapes $\rightarrow$ objects).
* **Attention:** A **"Focus" mechanism** — learn what parts of the input to look at.
* **Backpropagation:** The process of **credit assignment** backward through the network layers.

---

## 🧰 11. Practical Frameworks & Tools

| Category | Key Tools |
| :--- | :--- |
| **ML Libraries** | scikit-learn, XGBoost, LightGBM |
| **DL Frameworks** | **PyTorch**, **TensorFlow**, Keras |
| **Visualization** | TensorBoard, Weights & Biases |
| **Data & Core** | NumPy, Pandas, Matplotlib, Hugging Face Datasets |

---

## 🔬 12. Cheat Summary for Quick Recall

1.  **Model = Function** $\rightarrow$ maps inputs to outputs.
2.  **Learning = Optimization** $\rightarrow$ minimize loss.
3.  **Generalization = Simplicity** $\rightarrow$ avoid overfitting (regularization is key).
4.  **Neural Networks** = Stacked Linear + Nonlinear layers.
5.  **Deep Learning** = **Feature learning**, not manual engineering.
6.  **Data Quality** **>** Model Complexity (Garbage In, Garbage Out).
7.  **Transformers** = Scalable sequence learners with self-attention.
8.  **Backprop + Gradient Descent** = The core training engine.
9.  **Evaluation** = Balance accuracy with fairness, interpretability.