""" Train and use a classifier of Good and Bad images

To train and test, generate some more text-and-image pairs, putting them in
the Good and Bad sub-directories of the /inputs/ directory.  Then run this
file: the classifier will train on the /examples/ pairs and test on the
/inputs/ pairs.
"""


import os

import cohere
from cohere.classify import Example

from lib.api_key import cohere_api_key
from lib.settings import EXAMPLES_DIRECTORY, INPUTS_DIRECTORY, \
    TEXT_MODEL_SIZE, VAL_NAMES

# Needed so can run files from /resources/ diretory
examples_directory = EXAMPLES_DIRECTORY if os.path.basename(
    os.getcwd()) != 'resources' else os.path.join('..', EXAMPLES_DIRECTORY)

co = cohere.Client(cohere_api_key)


def read_examples():
    """
    Read examples from Good and Bad sub-directories of EXAMPLES_DIRECTORY

    :return: examples, List[Tuple[str, str]], each tuple (text, label value)
    """
    examples = list()
    for val_name in VAL_NAMES:
        val_dir = os.path.join(examples_directory, str(val_name))
        examples.extend(
            [(file.replace('_', ' ')[:-4], val_name)
             for file in os.listdir(val_dir) if file != '.gitkeep'])
    return examples


def read_inputs_from_pictures():
    """
    Read text-and-image pairs

    Assumes that there text-and-image pair files in the Good and Bad
    sub-directories of INPUTS_DIRECTORY

    :return:
        inputs: the text parts of the pairs, List[str]
        answers: the label value for each text, Dict[str: str]
    """
    inputs = list()
    answers = dict()
    for val_name in VAL_NAMES:
        val_dir = os.path.join(INPUTS_DIRECTORY, str(val_name))
        for file in os.listdir(val_dir):
            if file == '.gitkeep':
                continue
            text = file.replace('_', ' ')[:-4]
            inputs.append(text)
            answers[text] = val_name
    return inputs, answers
        
        
def classifier(examples, inputs):
    """
    Classify the inputs based on the examples

    :param examples: List[Tuple[str, str]]
    :param inputs: List[str]
    :return:
    """
    return co.classify(
      model=TEXT_MODEL_SIZE,
      inputs=inputs,
      examples=[Example(text, val) for text, val in examples])


def print_response(response, answers):
    """
    Print and analyse the response

    :param response: the cohere classifier response object
    :param answers: the label value for each text, Dict[str: str]
    :return: counts of scores from the classifier, Dict[str, in]
    """
    counts = {
        'AGREE': 0, 'DISAGREE': 0,
        'Bad allowed': 0, 'Good stopped': 0,
        'Bad': 0, 'Good': 0}
    for elt in response:
        print(elt.input)
        prediction = elt.prediction
        answer = answers[elt.input]
        counts[answer] += 1
        mark = 'AGREE' if prediction == answer else 'DISAGREE'
        if mark == 'DISAGREE':
            if answer == 'Bad':
                counts['Bad allowed'] += 1
            else:
                counts['Good stopped'] += 1
        counts[mark] += 1
        print(f'{mark:8}\t\t{prediction:4}\t\t'
              f'{elt.labels[elt.prediction].confidence:.2f} confidence\t\t'
              f'Human answer: {answer}')
        print()

    print(f'\nOverall:\n'
          f'AGREE {counts["AGREE"]}\tDISAGREE {counts["DISAGREE"]}\n'
          f'Bad  allowed {counts["Bad allowed"]:2} of {counts["Bad"]}\n'
          f'Good stopped {counts["Good stopped"]:2} of {counts["Good"]}')
    return counts


if __name__ == '__main__':
    examples = read_examples()
    inputs, answers = read_inputs_from_pictures()
    response = classifier(examples, inputs)
    print_response(response, answers)
