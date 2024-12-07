import pandas as pd
import matplotlib.pyplot as plt


def visualize_bought(
    user_id: str, 
    df: pd.DataFrame, 
    image_ids: pd.DataFrame
) -> None:
    '''
    Visualize bought items for a user.
    '''
    user_history = df[df['customer_id'] == user_id]
    num_transactions = len(user_history)
    col = 10
    rows = (num_transactions // col) + 1
    fig, axs = plt.subplots(rows, col, figsize=(col * 2, rows * 2))
    images = image_ids[image_ids['image_id'].isin(user_history['article_id'])]['path'].values
    for i, ax in enumerate(axs.flatten()):
        if i < len(images):
            image = plt.imread(images[i])
            ax.imshow(image)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_candidates(
    candidates: pd.DataFrame, 
    image_ids: pd.DataFrame
) -> None:
    '''
    Visualize candidate items for the user.
    '''
    num_candidates = len(candidates)
    col = 10
    rows = (num_candidates // col) + 1
    fig, axs = plt.subplots(rows, col, figsize=(col * 2, rows * 2))
    for i, ax in enumerate(axs.flatten()):
        if i < len(candidates):
            try:
                image_id = candidates.iloc[i]['article_id']
                image_path = image_ids[image_ids['image_id'] == image_id]['path'].values[0]
                image = plt.imread(image_path)
                ax.imshow(image)
            except IndexError:
                pass
        ax.axis('off')
    plt.tight_layout()
    plt.show()