# Shopee Product Matching - Kaggle Competition

This repository contains my solution for the [Shopee - Product Matching Kaggle Competition](https://www.kaggle.com/c/shopee-product-matching). The primary goal was to identify which products in a massive dataset were duplicates of each other, using only product titles and images. This project demonstrates a complete machine learning pipeline, from multi-modal deep learning to efficient similarity search. I used this project to gain further insight into the ML pipeline process as well as transformer architectures and text-based models.

**Final Result: Mean F1 Score of 0.577 on the private leaderboard.**
![Results](result.png)
---

### 🖼️ Visual Demonstration

My solution follows a two-stage process: first, generating a unified embedding from product data using a multi-modal model, and second, using these embeddings to find and group similar items.

#### Model Architecture

The core of the project is a neural network that fuses visual and textual information into a single vector.

```
+--------------------------+      +--------------------------------+
|      Product Image       |      |          Product Title         |
+--------------------------+      +--------------------------------+
             |                                   |
             v                                   v
+--------------------------+      +--------------------------------+
| Vision Transformer (ViT) |      |         BERT Encoder           |
|   (Image Embeddings)     |      |       (Text Embeddings)        |
+--------------------------+      +--------------------------------+
             |                                   |
             +----------------+------------------+
                              |
                              v
                 +--------------------------+
                 |  Concatenate & Project   |
                 |      (Fusion Layer)      |
                 +--------------------------+
                              |
                              v
                 +--------------------------+
                 | L2 Normalised Embedding  |
                 |      (512-dim Vector)    |
                 +--------------------------+
```

#### Example Output

The model generates L2-normalised embeddings for products. Similarity between products is measured by cosine similarity of these embeddings to find the closest matches.

| Query Product | Match 1 | Match 2 | Match 3 |
| :---: | :---: | :---: | :---: |
| ![Query](https://placehold.co/200x200/DBEAFE/3B82F6?text=Query+Item) | ![Match 1](https://placehold.co/200x200/DBEAFE/3B82F6?text=Match+1) | ![Match 2](https://placehold.co/200x200/DBEAFE/3B82F6?text=Match+2) | ![Match 3](https://placehold.co/200x200/DBEAFE/3B82F6?text=Match+3) |
| *Original Item* | *Top Similar Match* | *Top Similar Match* | *Top Similar Match* |
---

### ⚙️ Technical Approach & Methodology

My approach is centered around creating a powerful, combined embedding for each product and then efficiently finding the nearest neighbors in this embedding space.

1.  **Multi-Modal Model Architecture**:
    * **Image Embeddings**: A pre-trained Vision Transformer (`vit_b_16`) processes product images to capture visual features. The final classification head is removed to extract a 768-dimension feature vector.
    * **Text Embeddings**: A pre-trained BERT model (`bert-base-uncased`) processes product titles to capture semantic meaning. The `[CLS]` token's output is used as the sentence embedding.
    * **Fusion**: The image and text feature vectors are projected to a common dimension (512), concatenated, and then passed through a final linear layer to create a single, fused embedding.
    * **Normalisation**: The final embedding is L2-normalised. This is a critical step that projects the vectors onto a hypersphere, making cosine similarity an effective and efficient metric for measuring distance.

2.  **Inference and Candidate Generation**:
    * The trained model is used to generate a 512-dimension embedding for every product in the test set.
    * To find potential matches for each product, I used `scikit-learn`'s `NearestNeighbors` model, configured with a `cosine` metric. This allows for a highly optimised search for the top 50 most similar items, avoiding a slow, brute-force comparison.

3.  **Grouping and Thresholding**:
    * For each product, its 50 nearest neighbors are considered as potential matches.
    * A **similarity threshold of 0.85** is applied to the cosine similarity scores. Only neighbors with a similarity score *above* this threshold are considered true matches. This step is crucial for balancing the precision and recall of the final groups. Note: this could likely be further optimised with more experimentation to find an optimal similarity threshold, which could be lower or higher than the value used.

---

### 🛠️ Tech Stack

* **Core Libraries**: Python, PyTorch, Hugging Face Transformers
* **Data Handling**: Pandas, NumPy
* **ML & Computer Vision**: scikit-learn, Albumentations, Pillow (PIL)
* **Development**: Jupyter Notebooks (for exploration), Git & GitHub (for version control)

---

### 📂 Project Structure

```
shopee-product-matching/
├── data/
│   └── (Download from Kaggle)
├── notebooks/
│   └── investigate.ipynb
│   └── submission.ipynb
├── src/
│   ├── data_loader.py
│   ├── model.py
│   └── engine.py
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

### 🚀 Setup and Usage

To reproduce the results, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/muanderson/shopee-product-matching.git
    cd shopee-product-matching
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Download the competition data:**
    - Download from the [Shopee Kaggle page](https://www.kaggle.com/c/shopee-product-matching/data).
    - Unzip and place the contents into the `data/` directory.

4.  **Train the model locally:**

    Run the `train.py` script with appropriate arguments. Example:

    ```bash
    python train.py --output_dir /path/to/save/models \
                    --image_dir /path/to/train_images \
                    --tabular_data_path /path/to/train.csv \
                    --image_size 224 \
                    --batch_size 64 \
                    --learning_rate 5e-4 \
                    --epochs 100 \
                    --patience 10 \
                    --n_splits 5 \
                    --embedding_size 512
    ```

    Adjust paths and hyperparameters as needed.

5.  **Evaluate and create submission:**

    - Use the Kaggle notebook available in the repository at `notebooks/submission.csv`.
    - Upload this notebook to the Shopee Product Matching competition page on Kaggle to generate your submission.

6.  **Pre-trained model weights:**

    The pre-trained weights used can be found here and used to reproduce results without retraining:

    [https://www.kaggle.com/datasets/muanderson/shopee-model-weights/data](https://www.kaggle.com/datasets/muanderson/shopee-model-weights/data)

---

### 📈 Future Improvements

While the current score is okay, several areas could be explored to further improve performance:

* **Hyperparameter Tuning**: Systematically tune the `similarity_threshold` on a validation set to find the optimal balance between precision and recall.
* **Graph-Based Grouping**: Implement a more robust grouping strategy by treating matches as a graph. Finding the "connected components" of this graph would ensure that if A matches B and B matches C, then A, B, and C are all correctly placed in the same group.
* **Fine-Tuning with ArcFace**: Fine-tune the end-to-end model on the Shopee training data using a metric learning loss function like ArcFace. This would train the model to create a more discriminative embedding space, making the final grouping much easier and more accurate.
