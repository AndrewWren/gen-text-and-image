# gen-text-and-image

## Use the [cohere](https://cohere.ai) text generation API and the [Replicate](https://replicate.com/about) image generation API together to generate text-and-image pairs

(c) Andrew Wren 2022. MIT Licence but note that to run you need API keys 
and to agree to comply with [cohere](https://cohere.ai) and [Replicate](https://replicate.com/about)'s terms.

## Description

This project uses a transformer text generator, trained with an appropriate 
prompt, to generate a very short piece of text.  This text is then used by 
a stable diffusion image generator to generate a corresponding image.  In 
my experience about two-thirds of the images are 'good' in terms of both image 
quality and fit to the text.  This is, of course, a subjective judgement! See 30 examples in the `\examples` directory, which I have split into 
`Good` and `Bad` sub-directories.

## How to use

(1) After cloning the repo, create a conda environment by running `conda env 
create -f environment.yml` in the repo root directory.

(2) Get API keys from [cohere](https://cohere.ai) and [Replicate](https://replicate.com/about). 
Enter them in `/lib/api_key_template.py` as 
indicated, and then rename it to `/lib/api_key_template.py`.  DO NOT add 
this file to git or otherwise share it as it now contains your API keys.  
NOTE that, depending on quantity of generations used, you may need to pay for 
usage of these API keys.

(3) Run the Python program `main.py` to generate ten text-and-image pairs 
which can then be found in the `/generate/` directory.  The number to generate 
can be altered by using `-n <number>`  as command line argument. 

## Things to try

The prompt used (`PROMPT` in `/lib/settings.py`) is important for the 
quality of the pairs generated.  A very simple prompt like `'Draw an 
artistic picture of'` generates more Bad pairs than Good pairs.  The prompt 
currently used has a rate of about 1 Bad to every 2 Good.  "Mileage may vary"; 
your views on Good and Bad may differ from mine!  Can you find further 
improved prompts?

Try also using more `cohere` tools.  I found that training a 
[classifier](https://docs.cohere.ai/text-classification) 
on Good and Bad examples, and then using this classification as a filter,  
helped poor prompts, but getting a good prompt was more effective.  Visualisation of
[embeddings](https://docs.cohere.ai/embedding-wiki/) 
associated with the text does not suggest a particularly close relationship
between these (mean over token) embeddings but maybe that will change if 
you improve the prompt further.   See `/resources/` for `classifier.py` and 
`embedder.py` which may help.
