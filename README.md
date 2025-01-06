# Recommender System
This project implements the candidate generation step of a recommender system, providing several different models for this purpose.

> **Note**: This implementation focuses solely on the candidate generation step. A complete recommender system typically consists of three main steps:
> 1. Candidate Generation (implemented in this project)
> 2. Ranking
> 3. Post-ranking (applying business rules)

## Installation
1. Clone the repository:
```bash
git clone https://github.com/DogRog/Recommender-System.git
```

2. Install dependencies:
```bash
cd Recommender-System
pip install -r requirements.txt
```

## Project Structure
- `Models/`: Contains implementation of candidate generation models
  - `collaborative_filtering.py`: Collaborative filtering models (ALS, SVD)
  - `content_based_filtering.py`: Content-based filtering model and Numerical content-based filtering model
  - `hybrid.py`: Hybrid recommender model (ALS + numerical_CBF)
- `configs/`: Configuration files for model training and evaluation
  - `train/`: Training configurations for each model
  - `evaluate/`: Evaluation configurations for each model
- `Utils/`: Utility functions
- `Results/`: Storage for model evaluation results
- `Weights/`: Storage for trained model weights

## Dataset Preparation
1. Download the dataset from [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/overview)

2. Run `dataset_optimization.ipynb` to prepare and optimize the dataset for training. This step is **required** before proceeding with any training or evaluation.

## Usage

### Model Training
To train a model, use:
```bash
python train.py --config configs/train/{model}.yaml
```

### Model Evaluation
The evaluation process is integrated with Weights & Biases (W&B) for experiment tracking. To evaluate a model:
```bash
python evaluate.py --config configs/evaluate/{model}.yaml
```
Results and metrics will be automatically logged to your W&B dashboard.

### Jupyter Notebooks
- `dataset_optimization.ipynb`: **Required first step** - Prepares and optimizes the dataset for training
- `model_testing.ipynb`: Visualizes recommendation results using product images, helping to qualitatively assess model performance

## Data
The dataset can be downloaded from the following link: [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/overview)

## Results
The results of model training and evaluation are stored in the `Results` directory and are automatically tracked using W&B.
