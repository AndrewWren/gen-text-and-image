""" Plots embeddings of text from text-and-image pairs in the examples
directory, distinguishing between Good (green) and Bad (red) points

Need to install matplotlib, numpy and scikit-learn to the conda environment
"""

import cohere
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from lib.api_key import cohere_api_key
from resources.classifier import read_examples
from lib.settings import TEXT_MODEL_SIZE, VAL_NAMES

co = cohere.Client(cohere_api_key)


def embed_and_visualise():
    """
    Plots points corresponding to test in text-and-image pairs
    """
    examples = read_examples()
    texts = [example[0] for example in examples]
    response = co.embed(
      model=TEXT_MODEL_SIZE,
      texts=texts)

    tsne = TSNE(perplexity=5)
    vis = tsne.fit_transform(np.vstack(response.embeddings))
    vis_dict = {name: [vis_instance
      for example, vis_instance in zip(examples, vis) if example[1] == name
    ] for name in VAL_NAMES}
    vis_dict = {key: np.vstack(value).T for key, value in vis_dict.items()}
    colours = {'Bad': 'r', 'Good': 'g'}
    plt.figure()
    for name, vis in vis_dict.items():
      plt.scatter(*vis, c=colours[name])
    plt.show()


if __name__ == '__main__':
    embed_and_visualise()
