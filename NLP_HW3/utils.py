import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    text = example["text"]
    words = word_tokenize(text)  # Tokenize the text
    transformed_words = []

    # Define parameters for the transformations
    synonym_prob = 0.2  # Probability to replace a word with a synonym
    typo_prob = 0.1     # Probability to introduce a typo in a word
    typo_map = {
        'a': 's', 's': 'a', 'd': 's', 'f': 'd', 'g': 'f',
        'q': 'w', 'w': 'q', 'e': 'w', 'r': 'e', 't': 'r',
        'y': 't', 'u': 'y', 'i': 'u', 'o': 'i', 'p': 'o',
        'z': 'x', 'x': 'z', 'c': 'x', 'v': 'c', 'b': 'v',
        'n': 'b', 'm': 'n', 'l': 'k', 'k': 'j', 'j': 'h'
    }

    for word in words:
        # Decide whether to replace with a synonym
        if random.random() < synonym_prob:
            synonyms = wordnet.synsets(word)
            if synonyms:
                # Find synonyms and choose one at random
                synonym_words = synonyms[0].lemma_names()
                synonym = random.choice(synonym_words)
                transformed_words.append(synonym.replace('_', ' '))
            else:
                transformed_words.append(word)
        # Decide whether to introduce a typo
        elif random.random() < typo_prob:
            typo_word = list(word)
            idx = random.randint(0, len(typo_word) - 1)
            if typo_word[idx].lower() in typo_map:
                typo_word[idx] = typo_map[typo_word[idx].lower()]
            transformed_words.append("".join(typo_word))
        else:
            transformed_words.append(word)

    # Detokenize the transformed words back into a sentence
    transformed_text = TreebankWordDetokenizer().detokenize(transformed_words)
    example["text"] = transformed_text
    return example
