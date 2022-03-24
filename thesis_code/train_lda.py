import logging
import os
import random
import numpy as np
import sys
from pathlib import Path
import re
import json
from pprint import pprint
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from nltk import WordNetLemmatizer, RegexpTokenizer
from transformers import set_seed


def load_wikitext(path, samples=100000):
    heading_pattern = '( \n [=\s].*[=\s] \n)'
    train_data = Path(f'{path}data_wikitext-103-raw/wiki.train.raw').read_text(encoding='utf-8')
    train_split = re.split(heading_pattern, train_data)
    train_headings = [x[7:-7] for x in train_split[1::2]]
    train_articles = [x for x in train_split[2::2]]
    return random.choices(train_articles, k=samples)


def load_arxiv(path, samples=100000):
    def get_metadata():
        with open(f'{path}data_arxiv-metadata-oai-snapshot.json', 'r') as f:
            for line in f:
                yield line

    metadata = get_metadata()
    size = 0
    for paper in metadata:
        size += 1
    choices = random.choices(list(np.arange(size)), k=samples)
    choices.sort()
    metadata = get_metadata()
    step = 0
    idx = 0
    corpus = []
    for paper in metadata:
        if idx >= samples:
            break
        if step == choices[idx]:
            if step != choices[idx+1]:
                step += 1
            corpus.append(json.loads(paper)['abstract'])
            idx += 1
        else:
            step += 1
    return corpus


def load_json_choices(filename, samples=100000):
    with open(filename, 'r') as file:
        train_articles = json.load(file)
    return random.choices(train_articles, k=samples)


def load_json(filename):
    with open(filename, 'r') as file:
        train_articles = json.load(file)
    return train_articles


def load_dataset(data_path, set_name, sampling_method):
    if sampling_method == "multinomial":
        sampling = ""
    else:
        sampling = "-" + sampling_method

    dataset = "dataset1"
    if set_name[-2:] == "_2":
        dataset = "dataset2"

    docs = None
    if set_name == "arxiv":
        if set_name[-2:] == "_2":
            set_seed(1337)
        docs = load_arxiv(data_path)
    elif set_name == "wiki_nt":
        if set_name[-2:] == "_2":
            set_seed(1337)
        docs = load_wikitext(data_path)
    elif set_name == "gpt2_nt":
        docs = load_json(f"{data_path}{dataset}-gpt2-wiki_nt{sampling}.json")
    elif set_name == "gpt2":
        docs = load_json(f"{data_path}{dataset}-gpt2{sampling}.json")

    return docs


def tokenize_text(docs, add_bigrams=True, add_trigrams=True):
    # Tokenize the documents.
    # Split the documents into tokens.
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()  # Convert to lowercase.
        docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 1] for doc in docs]

    # Lemmatize the documents. Better than stemmer as is easier to read
    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

    # Add bigrams to docs (only ones that appear 20 times or more).
    bigram = Phrases(docs, min_count=5, threshold=10.0) if add_bigrams or add_trigrams else None

    trigram = Phrases(bigram[docs], min_count=5, threshold=10.0) if add_trigrams else None

    if add_bigrams or add_trigrams:
        for idx in range(len(docs)):
            doc = docs[idx]
            bigrams = list()
            trigrams = list()
            if add_bigrams:
                for token in bigram[doc]:
                    if '_' in token:
                        # Token is a bigram, add to document.
                        bigrams.append(token)
            if add_trigrams:
                for token in trigram[bigram[doc]]:
                    cnt = token.count('_')
                    if cnt == 2:
                        # Token is a trigram, add to document.
                        trigrams.append(token)

            if add_bigrams:
                for token in bigrams:
                    docs[idx].append(token)

            if add_trigrams:
                for token in trigrams:
                    docs[idx].append(token)

    return docs


def tokenize_create_dictionary(docs):
    # Remove rare and common tokens.
    # Create a dictionary representation of the documents.
    dictionary = Dictionary(docs)

    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    dictionary.filter_extremes(no_below=20, no_above=0.5)

    return docs, dictionary


def tokenize_bow_single(docs):
    docs = tokenize_text(docs, add_bigrams=True, add_trigrams=True)
    docs, dictionary = tokenize_create_dictionary(docs)

    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    print('>> Number of unique tokens: %d' % len(dictionary))
    print('>> Number of documents: %d' % len(corpus))

    return dictionary, corpus


def tokenize_bow_dual(docs0, docs1, union=False):  # False is intersection of dictionaries
    docs_new_0 = tokenize_text(docs0, add_bigrams=True, add_trigrams=True)
    docs_new_1 = tokenize_text(docs1, add_bigrams=True, add_trigrams=True)
    docs_new_0, dic0 = tokenize_create_dictionary(docs_new_0)
    docs_new_1, dic1 = tokenize_create_dictionary(docs_new_1)

    if union:
        transformer = dic0.merge_with(dic1)
    else:
        good_ids0 = []
        good_ids1 = []
        for good_value in set(dic0.values()).intersection(set(dic1.values())):
            good_ids0.append(dic0.token2id[good_value])
            good_ids1.append(dic1.token2id[good_value])
        dic0.filter_tokens(good_ids=good_ids0)

    dic1 = dic0

    # Bag-of-words representation of the documents.
    cor0 = [dic0.doc2bow(doc) for doc in docs0]
    cor1 = [dic1.doc2bow(doc) for doc in docs1]
    print('>> Number of unique tokens in dic0: %d' % len(dic0))
    print('>> Number of unique tokens in dic1: %d' % len(dic1))
    print('>> Number of documents of cor0: %d' % len(cor0))
    print('>> Number of documents of cor1: %d' % len(cor1))

    return dic0, cor0, dic1, cor1


def train_lda(dictionary, corpus, topics):
    # Train LDA model.
    # TODO: Add neural topic model

    # Set training parameters.
    num_topics = topics
    chunksize = 1000000
    passes = 30
    iterations = 500
    eval_every = 1  # Don't evaluate model perplexity, takes too much time.

    # Make a index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    model = LdaMulticore(
        corpus=corpus,
        num_topics=num_topics,
        id2word=id2word,
        workers=48,
        chunksize=chunksize,
        passes=passes,
        alpha='symmetric',
        eta='auto',
        eval_every=eval_every,
        iterations=iterations,
    )

    top_topics = model.top_topics(corpus)

    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('>> Average topic coherence: %.4f.' % avg_topic_coherence)

    pprint(top_topics)

    return model


def main():
    """
    Command:
        python train_lda.py [data_path] [first corpus] [second corpus] [focus] [sampling method] [number of topics] [merge technique] [variance index]

        Corpus Options (always use gpt2_nt, gpt2 or wiki first, in that order (convention)):
            gpt2_nt, gpt2, wiki_nt, arxiv
        Focus options:
            first: First corpus is used for lda model creation
            second: Second corpus is used for lda model creation
        Sampling method:
            multinomial, typ_p, top_p
        Topic options:
            2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, ...
        Merge technique:
            union: Unionize dictionaries
            intersection: Intersect dictionaries
        Variance index:
            Model index when calculating the variance (changes the seed)

    Examples:
        python train_lda.py /cluster/work/cotterell/knobelf/data/ gpt2_nt wiki_nt first multinomial 5 union
        python train_lda.py /cluster/work/cotterell/knobelf/data/ gpt2_nt gpt2 first typ_p 10 intersection 1
        python train_lda.py /cluster/work/cotterell/knobelf/data/ gpt2_nt gpt2 second typ_p 10 intersection 2
        python train_lda.py /cluster/work/cotterell/knobelf/data/ gpt2 arxiv first typ_p 10 intersection 3
        python train_lda.py /cluster/work/cotterell/knobelf/data/ wiki_nt arxiv first typ_p 10 intersection 4

    """
    if len(sys.argv) < 8:
        print(">> ERROR: Incorrect number of input arguments")
        return

    data_path = sys.argv[1]
    first = sys.argv[2]
    second = sys.argv[3]
    focus = sys.argv[4]
    sampling = sys.argv[5]
    num_topics = int(sys.argv[6])
    combi = sys.argv[7]

    if data_path[-1] != "/":
        data_path += "/"

    if first not in ["gpt2", "gpt2_nt", "wiki_nt", "arxiv"]:
        print(">> ERROR: undefined first input")
        return

    if second not in ["gpt2", "gpt2_nt", "wiki_nt", "arxiv"]:
        print(">> ERROR: undefined second input")
        return

    if sampling not in ["multinomial", "typ_p", "top_p"]:
        print(">> ERROR: undefined sampling input")
        return

    folder_name = f"lda-{first}-{second}"

    if sampling != "multinomial":
        folder_name += "-" + sampling

    if first == second:
        first += "_1"
        second += "_2"

    if num_topics <= 1:
        print(">> ERROR: undefined num_topics input")
        return

    if combi == "intersection":
        union = False
    elif combi == "union":
        union = True
    else:
        print(">> ERROR: undefined combi input")
        return

    set_seed(42)
    docs1 = load_dataset(data_path, first, sampling)
    docs2 = load_dataset(data_path, second, sampling)
    set_seed(42)

    dic1, cor1, dic2, cor2 = tokenize_bow_dual(docs1, docs2, union)

    if focus == "first":
        model_name = first
        dictionary = dic1
        corpus = cor1
    elif focus == "second":
        model_name = second
        dictionary = dic2
        corpus = cor2
    else:
        print(">> ERROR: undefined focus input")
        return

    index = ""
    if len(sys.argv) == 9:
        index = "/" + sys.argv[8]
        seed = 42 + 7**int(index[1:])
        set_seed(seed)
        print(f">> SEED changing to: {seed}")

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    model = train_lda(dictionary, corpus, num_topics)

    file_path = f"{data_path}{folder_name}/{model_name}{index}/{combi}/{num_topics}/"

    os.makedirs(os.path.dirname(f"{file_path}corpus_{num_topics}"), exist_ok=True)
    with open(f"{file_path}corpus_{num_topics}", "w") as file:
        json.dump(corpus, file, indent=2)
    dictionary.save(f"{file_path}dictionary_{num_topics}")
    model.save(f"{file_path}ldamodel_{num_topics}")


if __name__ == "__main__":
    main()
