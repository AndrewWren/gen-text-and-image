import os


PROMPT = """This is a list of image captions:

1. Tiger in the snow
2. View out of window
3. Cat in a stetson
4. Mountainous landscape
5. Sunset over the ocean"""

EXAMPLES_DIRECTORY = 'examples'
GENERATED_DIRECTORY = 'generated'
INPUTS_DIRECTORY = 'inputs'

VAL_NAMES = ['Good', 'Bad']

TEXT_MODEL_SIZE = 'large'
TEXT_MODEL_TEMPERATURE = 1.
SAVE_PROMPT = False

if SAVE_PROMPT:
    path = os.path.join(GENERATED_DIRECTORY, 'prompt.txt')
    if os.path.exists(path):
        exit(f'prompt.txt file already exists as {path}')
    with open(path, 'w') as f:
        f.write(PROMPT)
    del path
