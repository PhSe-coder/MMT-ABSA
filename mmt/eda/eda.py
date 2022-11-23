# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

import random
import re
from typing import List, NamedTuple, Tuple

import nltk
from nltk.corpus import wordnet

__all__ = [
    'eda', 'Word'
    ]

# for the first time you should download wordnet
nltk.download('wordnet')
random.seed(1)

class Word(NamedTuple):
    token: str
    gold_label: str
    hard_label: str

# stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
   'ours', 'ourselves', 'you', 'your', 'yours',
   'yourself', 'yourselves', 'he', 'him', 'his',
   'himself', 'she', 'her', 'hers', 'herself',
   'it', 'its', 'itself', 'they', 'them', 'their',
   'theirs', 'themselves', 'what', 'which', 'who',
   'whom', 'this', 'that', 'these', 'those', 'am',
   'is', 'are', 'was', 'were', 'be', 'been', 'being',
   'have', 'has', 'had', 'having', 'do', 'does', 'did',
   'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
   'because', 'as', 'until', 'while', 'of', 'at',
   'by', 'for', 'with', 'about', 'against', 'between',
   'into', 'through', 'during', 'before', 'after',
   'above', 'below', 'to', 'from', 'up', 'down', 'in',
   'out', 'on', 'off', 'over', 'under', 'again',
   'further', 'then', 'once', 'here', 'there', 'when',
   'where', 'why', 'how', 'all', 'any', 'both', 'each',
   'few', 'more', 'most', 'other', 'some', 'such', 'no',
   'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
   'very', 's', 't', 'can', 'will', 'just', 'don',
   'should', 'now', '']

# cleaning up text
def get_only_chars(line):

    clean_line = ""

    line = line.replace("’", "'")
    line = line.replace("'", "")
    line = line.replace("-", " ") # replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line

########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################

def synonym_replacement(words: Tuple[Word], n: int):
    new_words = list(words)
    random_word_list = list(set(word for word in words if word.token not in stop_words))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word.token)
        if len(synonyms) >= 1:
            old_words: List[Word] = []
            for word in new_words:
                if word == random_word:
                    synonym = random.choice(synonyms)
                    for syn in synonym.split(' '):
                        old_words.append(Word(syn, *list(word)[1:]))
                else:
                    old_words.append(word)
            new_words = old_words.copy()
            # print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n: # only replace up to n words
            break
    return new_words

def get_synonyms(word: str) -> List[str]:
    synonyms = set()
    for syn in wordnet.synsets(word.lower()):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm']).strip()
            if synonym == '':
                continue
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words: Tuple[Word], p: float):
    # obviously, if there's only one word, don't delete it
    if len(words) <= 1:
        return list(words)
    # randomly delete words with probability p
    new_words = [word for word in words if random.uniform(0, 1) > p]
    # if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        new_words.append(random.choice(words))
    return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words: Tuple[Word], n: int):
    new_words = list(words)
    for _ in range(n):
        swap_word(new_words)
    return new_words

def swap_word(words: List[Word]):
    random_idx_1 = random.randint(0, len(words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(words)-1)
        counter += 1
        if counter > 3:
            return
    words[random_idx_1], words[random_idx_2] = words[random_idx_2], words[random_idx_1]

########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words: Tuple[Word], n: int):
    new_words = list(words)
    for _ in range(n):
        add_word(new_words)
    return new_words

def add_word(new_words: List[Word]):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = random.choice(new_words)
        synonyms = get_synonyms(random_word.token)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    random_synonym = random_synonym.split(' ')
    # synonyms with multi-word
    while len(random_synonym) != 0:
        new_words.insert(random_idx, Word(random_synonym[-1], *list(random_word)[1:]))
        random_synonym = random_synonym[:-1]

########################################################################
# main data augmentation function
########################################################################

def eda(words: Tuple[Word], alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9, sep="***"):
    num_words = len(words)
    augmented_sentences = []
    num_new_per_technique = int(num_aug/4)+1
    # sr
    if (alpha_sr > 0):
        n_sr = max(1, int(alpha_sr*num_words))
        i = 0
        iters = 0
        while i < num_aug:
            iters += 1
            a_words = synonym_replacement(words, n_sr)
            result = (' '.join(getattr(word, field) for word in a_words) for field in Word._fields)
            text = sep.join(result)
            if iters == 100:
                root_text = sep.join(' '.join(getattr(word, field) for word in words)
                                     for field in Word._fields)
                augmented_sentences.append(root_text)
                break
            if text not in augmented_sentences:
                augmented_sentences.append(text)
                i += 1
                iters = 0
        if len(augmented_sentences) % 2 != 0:
            augmented_sentences.pop()

# ri
    if (alpha_ri > 0):
        n_ri = max(1, int(alpha_ri*num_words))
        for _ in range(num_new_per_technique):
            a_words = random_insertion(words, n_ri)
            result = (' '.join(getattr(word, field) for word in a_words) for field in Word._fields)
            augmented_sentences.append(sep.join(result))

# rs
    if (alpha_rs > 0):
        n_rs = max(1, int(alpha_rs*num_words))
        for _ in range(num_new_per_technique):
            a_words = random_swap(words, n_rs)
            result = (' '.join(getattr(word, field) for word in a_words) for field in Word._fields)
            augmented_sentences.append(sep.join(result))

# rd
    if (p_rd > 0):
        for _ in range(num_new_per_technique):
            a_words = random_deletion(words, p_rd)
            result = (' '.join(getattr(word, field) for word in a_words) for field in Word._fields)
            augmented_sentences.append(sep.join(result))

# augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
    random.shuffle(augmented_sentences)

    # trim so that we have the desired number of augmented sentences
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

# append the original sentence
# sentence = [word.token for word in words]
# labels = [word.label for word in words]
# augmented_sentences.append(f"{' '.join(sentence)}***{' '.join(labels)}")
    return augmented_sentences
