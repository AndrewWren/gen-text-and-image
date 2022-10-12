""" Main file to generate text and images

Run to generate number of text-image pairs specified in call on generate at
foot of file.  Will save this images, with text as save name, in /generated
directory
"""

import argparse
import os
import re

import cohere
import replicate
import requests

from lib.api_key import cohere_api_key, replicate_api_key
from lib.settings import GENERATED_DIRECTORY, PROMPT, TEXT_MODEL_SIZE,\
    TEXT_MODEL_TEMPERATURE


co = cohere.Client(cohere_api_key)
rep = replicate.Client(api_token=replicate_api_key)
image_model = rep.models.get("stability-ai/stable-diffusion")
re_numbering = re.compile(r'^\d*\. ')


def text_generator():
    """
    Use the cohere text generator to produce texts based on PROMPT

    :return: texts, List[str]
    """
    response = co.generate(
        model=TEXT_MODEL_SIZE,
        prompt=PROMPT,
        max_tokens=20,
        temperature=TEXT_MODEL_TEMPERATURE,
        k=0,
        p=0.75,
        frequency_penalty=0,
        presence_penalty=0,
        stop_sequences=[],
        return_likelihoods='NONE',
    )
    resp = response.generations[0].text.lstrip().rstrip()
    texts = list()
    for elt in resp.split('\n'):
        spl = re_numbering.split(elt)
        if len(spl) < 2:
            continue
        text = spl[1]
        if (('_' in text) or ('/' in text)):  # as these screw up save names
            continue
        texts.append(text)
    return texts


def picture_generator(texts, max_to_do=1_000_000):
    """
    Generate pictures from captions listed in texts

    :param texts: List[str]
    :param max_to_do:  maximum number of pictures to generate, int
    :return: number of pictures generated, n_done, int
    """
    n_done = 0
    for text in texts:
        if n_done >= max_to_do:
            return n_done
        # noinspection PyUnresolvedReferences
        try:  # the replicate image generated can throw an NSFW error
            image = image_model.predict(prompt=text)
        except replicate.exceptions.ModelError:
            continue
        n_done += 1
        path = os.path.join(
            GENERATED_DIRECTORY,
            f'{text.replace(" ", "_")}.jpg'
        )
        with open(path, 'wb') as handle:
            response = requests.get(image[0], stream=True)
            if not response.ok:
                print(response)
            for block in response.iter_content(1024):
                if not block:
                    break
                handle.write(block)
        print(text)

    return n_done


def generate(n_to_generate):
    """
    Generate text and image pairs, saving the images in <text>.jpg files

    :param n_to_generate: number of pairs to generate, int
    """
    number_generated = 0
    while number_generated < n_to_generate:
        texts = text_generator()
        number_generated += picture_generator(
            texts, max_to_do=n_to_generate - number_generated)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n', default=10, help='How many text-and-image pairs to generate')
    args = parser.parse_args()
    generate(int(args.n))
