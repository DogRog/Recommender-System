# Recommender System

This project implements several different recommender system models.

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

## Usage

The project includes several Jupyter notebooks for data exploration, model training, and evaluation:

* `eda.ipynb`: Exploratory Data Analysis
* `model_testing.ipynb`: Model training and testing
* `dataset optimization.ipynb`: Optimizing the dataset
* `playground.ipynb`: A playground for experimenting with different models and parameters.

To evaluate the models, run the following command:
```bash
python evaluate.py --confi
```


The implemented models are located in the `Models` directory:

* `collaborative_filtering.py`: Collaborative filtering models (ALS, SVD).
* `content_based_filtering.py`: Content-based filtering model.
* `numerical_CBF.py`: Numerical content-based filtering model.
* `hybrid.py`: Hybrid recommender model. (ALS + numerical_CBF)

Utility functions are located in the `Utils` directory.


## Data

The dataset can be downloaded from the following link: [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/overview)

## Results

The results of model training and evaluation are stored in the `Results` directory.

## Weights

Trained model weights are stored in the `Weights` directory.


## Contributing
