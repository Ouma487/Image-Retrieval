import os 
import numpy as np 
from sklearn.manifold import SpectralEmbedding, Isomap
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from utils_features import compose_features, load_depth_estimator
import matplotlib.pyplot as plt
import argparse


def pixels_to_nodes(features, params):
    """Converts the pixels into nodes for graph embedding, potentially grouping small blocks of pixels into superpixels for computational efficiency. 

    Args:
        features (ndarray): Stacked image transformations chosen as features.
        params (dict): Parameters for the pixel-to-nodes transformation:
                        - 'pos_weight': Weight assigned to the positional arguments.
                        - 'super_pixel': Size of the superpixel.

    Returns:
        s_pixels, s_pixels_indexes: Graph nodes.
    """
    
    pixels = [[i * params['pos_weight'], j * params['pos_weight']] + list(features[i, j]) for i in range(features.shape[0]) for j in range(features.shape[1])]
    pixels = np.array(pixels)
    
    s_pixels = []
    i, j = 0, 0
    
    s_pixels_indexes = []
    
    while i < features.shape[0]:
        while j < features.shape[1]: 
            indexes = [(i + s_i) * (features.shape[0]) + j + s_j for s_i in range(params['super_pixel']) for s_j in range(params['super_pixel'])]
            s_pixels.append(pixels[indexes].flatten())
            s_pixels_indexes.append(indexes)
            j += params['super_pixel'] 
        j = 0
        i += params['super_pixel']
      
    s_pixels = np.array(s_pixels) 
    return s_pixels , s_pixels_indexes
    
def construct_embedding(nodes, params): 
    """Construct the graph and project the data into a lower-dimensional space.

    Args:
        nodes (ndarray): Graph nodes.
        params (dict): Parameters of the spectral embedding:
                        - 'embedding_method': Manifold learning method ('spectral' or 'isomap').
                        - 'n_components': Dimension of the projection space.
                        - 'n_neighbors': Number of neighbors used for the graph construction.

    Returns:
        ndarray: Projected data.
    """

    if params['embedding_method'] == 'spectral': 
        model = SpectralEmbedding(n_components=params['n_components'], n_neighbors=params['n_neighbors'])
    elif params['embedding_method'] == 'isomap': 
        model = Isomap(n_components=params['n_components'], n_neighbors=params['n_neighbors'])
        
    X_proj = model.fit_transform(nodes) 
    
    return X_proj

def cluster_data(embedding, params): 
    """Cluster the data into k clusters (image segments) using the given embedding.

    Args:
        embedding (ndarray): Projected pixels.
        params (dict): Parameters of the clustering:
                        - 'cluster_method': Clustering method ('knn' or 'mixture').
                        - 'n_clusters': Number of clusters.

    Returns:
        list: List of labels of each projected node.
    """
    
    if params['cluster_method'] == 'knn': 
        model = KMeans(n_clusters=params['n_clusters']).fit(embedding)
        labels = model.labels_
    elif params['cluster_method'] == 'mixture': 
        model = GaussianMixture(n_components=params['n_clusters'], random_state=0).fit(embedding)
        labels = model.predict(embedding)
        
    return labels 

def segment_image(features, params={}): 
    """Main function that performs segmentation by grouping different blocks.

    Args:
        features (ndarray): Stacked image transformations chosen as features.
        params (dict): Parameters of the segmentation:
                        - 'pos_weight': Weight assigned to the positional arguments.
                        - 'super_pixel': Size of the superpixel.
                        - 'embedding_method': Manifold learning method ('spectral' or 'isomap').
                        - 'n_components': Dimension of the projection space.
                        - 'n_neighbors': Number of neighbors used for the graph construction.
                        - 'cluster_method': Clustering method ('knn' or 'mixture').
                        - 'n_clusters': Number of clusters.

    Returns:
        labels_shaped: Mask with the same shape as the image where each pixel refers to a cluster.
        embedding : Projected data.
        labels : 1D array of all the labels.
    """

    nodes, indexes = pixels_to_nodes(features, params)
    
    embedding = construct_embedding(nodes, params)
    
    labels = cluster_data(embedding, params)
    
    labels_shaped = np.zeros(features.shape[0] * features.shape[1])
    for i in range(len(labels)): 
        labels_shaped[indexes[i]] = labels[i]
        
    labels_shaped = labels_shaped.reshape(features.shape[0], features.shape[1])
    return labels_shaped, embedding, labels


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Unsupervised Segmentation')
    parser.add_argument('--input', metavar='FILENAME',
                    help='input image file name', required=True)
    parser.add_argument('--output', metavar='FILENAME',
                    help='output image file name', required=True)
    args = parser.parse_args()
    
    estimator = load_depth_estimator()
    
    #TODO: Modify the features weights to adapt to image domain
    features, t = compose_features(args.input, estimator, shape=(256, 256), weights=[0.8, 1.2, 0.8, 0.1])
    
    #TODO: Modify the spectral embedding parameters to adapt to image domain
    params = {'super_pixel':2, 'embedding_method':'spectral', 'n_components':10, 'n_neighbors':50, 'cluster_method':'knn', 'n_clusters':7, 'pos_weight':0.4}
    labels_shaped, embedding, labels = segment_image(features, params=params)
    
    fig, axes = plt.subplots(1, len(t)+1, figsize=(16, 4))

    for i in range(len(t)) : 
        axes[i].imshow(t[i])
        axes[i].axis('off')
        axes[i].set_title(f'Feature {i}')
        
    axes[-1].imshow(labels_shaped)
    axes[-1].axis('off')
    axes[-1].set_title(f'Segmentation')
    
    plt.savefig(args.output)
    # plt.show()
