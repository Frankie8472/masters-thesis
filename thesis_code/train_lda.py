import gc
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


def load_wikitext(samples=100000):
    heading_pattern = '( \n [=\s].*[=\s] \n)'
    train_data = Path('/cluster/work/cotterell/knobelf/data/data_wikitext-103-raw/wiki.train.raw').read_text(encoding='utf-8')
    train_split = re.split(heading_pattern, train_data)
    train_headings = [x[7:-7] for x in train_split[1::2]]
    train_articles = [x for x in train_split[2::2]]
    return random.choices(train_articles, k=samples)


def load_arxiv(samples=100000):
    def get_metadata():
        with open('/cluster/work/cotterell/knobelf/data/data_arxiv-metadata-oai-snapshot.json', 'r') as f:
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


def tokenize(docs):
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
    bigram = Phrases(docs, min_count=20)
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)

    # Remove rare and common tokens.
    # Create a dictionary representation of the documents.
    dictionary = Dictionary(docs)

    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    dictionary.filter_extremes(no_below=20, no_above=0.5)

    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    return dictionary, corpus


def tokenize_special(docs0, docs1, union=False):  # False is intersection
    # Tokenize the documents.
    # Split the documents into tokens.
    tokenizer = RegexpTokenizer(r'\w+')

    for idx in range(len(docs0)):
        docs0[idx] = docs0[idx].lower()  # Convert to lowercase.
        docs0[idx] = tokenizer.tokenize(docs0[idx])  # Split into words.
    for idx in range(len(docs1)):
        docs1[idx] = docs1[idx].lower()  # Convert to lowercase.
        docs1[idx] = tokenizer.tokenize(docs1[idx])  # Split into words.

    # Remove numbers, but not words that contain numbers.
    docs0 = [[token for token in doc if not token.isnumeric()] for doc in docs0]
    docs1 = [[token for token in doc if not token.isnumeric()] for doc in docs1]

    # Remove words that are only one character.
    docs0 = [[token for token in doc if len(token) > 1] for doc in docs0]
    docs1 = [[token for token in doc if len(token) > 1] for doc in docs1]

    # Lemmatize the documents. Better than stemmer as is easier to read
    lemmatizer = WordNetLemmatizer()
    docs0 = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs0]
    docs1 = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs1]

    # Add bigrams to docs (only ones that appear 20 times or more).
    bigram = Phrases(docs0, min_count=20)
    for idx in range(len(docs0)):
        for token in bigram[docs0[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs0[idx].append(token)
    bigram = Phrases(docs1, min_count=20)
    for idx in range(len(docs1)):
        for token in bigram[docs1[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs1[idx].append(token)

    # Remove rare and common tokens.
    # Create a dictionary representation of the documents.
    dic0 = Dictionary(docs0)
    dic1 = Dictionary(docs1)

    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    dic0.filter_extremes(no_below=20, no_above=0.5)
    dic1.filter_extremes(no_below=20, no_above=0.5)

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
    print('Number of unique tokens in dic0: %d' % len(dic0))
    print('Number of unique tokens in dic1: %d' % len(dic1))
    print('Number of documents of cor0: %d' % len(cor0))
    print('Number of documents of cor1: %d' % len(cor1))

    return dic0, cor0, dic1, cor1


def train_lda(dictionary, corpus, topics):
    # Train LDA model.

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
    print('Average topic coherence: %.4f.' % avg_topic_coherence)

    pprint(top_topics)

    return model


def main():
    if len(sys.argv) < 4:
        print("ERROR: Incorrect number of input arguments")
        return

    data_folder = "/cluster/work/cotterell/knobelf/data/"

    union = bool(int(sys.argv[1]))
    dataset = int(sys.argv[2])
    topics = int(sys.argv[3])
    pathname = None

    combi = "union" if union else "intersection"

    extra_folder = ""
    iteration = None
    if len(sys.argv) >= 5:
        iteration = int(sys.argv[4])
        extra_folder = "/" + sys.argv[4]

    seed = seed2 = 42
    if iteration is not None:
        seed2 = 42 + 7**iteration
        if iteration < 6:
            seed = seed2

    print(f"SEED: {seed}")
    print(f"SEED2: {seed2}")

    if not len(sys.argv) >= 6 and dataset == 0:
        print("ERROR: no path argument and dataset == 0")
    elif len(sys.argv) >= 6:
        pathname = sys.argv[5]

    set_seed(seed)
    samples = 100000

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    if dataset == 0:
        data_train = load_json(pathname)
        dictionary, corpus = tokenize(data_train)
        model = train_lda(dictionary, corpus, topics)
        os.makedirs(os.path.dirname(f"{data_folder}test/corpus2_{topics}"), exist_ok=True)
        with open(f"{data_folder}test/corpus2_{topics}", "w") as file:
            json.dump(corpus, file, indent=2)
        dictionary.save(f"{data_folder}test/dictionary2_{topics}")
        model.save(f"{data_folder}test/ldamodel2_{topics}")

    # Create LDA Model for wiki_nt and gpt2_nt
    elif dataset == 1 or dataset == 2:
        data_train_1 = load_wikitext(samples)
        data_train_2 = load_json(f"{data_folder}dataset1-gpt2-wiki_nt.json")
        dictionary_1, corpus_1, dictionary_2, corpus_2 = tokenize_special(data_train_1, data_train_2, union)

        set_seed(seed2)
        if dataset == 1:
            model_1 = train_lda(dictionary_1, corpus_1, topics)

            os.makedirs(os.path.dirname(f"{data_folder}lda-gpt2_nt-wiki_nt/wiki_nt{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                        exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-wiki_nt/wiki_nt{extra_folder}/{combi}/{topics}/corpus_{topics}", "w") as file:
                json.dump(corpus_1, file, indent=2)
            dictionary_1.save(f"{data_folder}lda-gpt2_nt-wiki_nt/wiki_nt{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_1.save(f"{data_folder}lda-gpt2_nt-wiki_nt/wiki_nt{extra_folder}/{combi}/{topics}/ldamodel_{topics}")
        else:
            model_2 = train_lda(dictionary_2, corpus_2, topics)

            os.makedirs(os.path.dirname(f"{data_folder}lda-gpt2_nt-wiki_nt/gpt2_nt{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                        exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-wiki_nt/gpt2_nt{extra_folder}/{combi}/{topics}/corpus_{topics}", "w") as file:
                json.dump(corpus_2, file, indent=2)
            dictionary_2.save(f"{data_folder}lda-gpt2_nt-wiki_nt/gpt2_nt{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_2.save(f"{data_folder}lda-gpt2_nt-wiki_nt/gpt2_nt{extra_folder}/{combi}/{topics}/ldamodel_{topics}")

    # Create LDA Model for gpt2_nt and gpt2_nt
    elif dataset == 3 or dataset == 4:
        data_train_1 = load_json(f"{data_folder}dataset1-gpt2-wiki_nt.json")
        data_train_2 = load_json(f"{data_folder}dataset2-gpt2-wiki_nt.json")
        dictionary_1, corpus_1, dictionary_2, corpus_2 = tokenize_special(data_train_1, data_train_2, union)

        set_seed(seed2)
        if dataset == 3:
            model_1 = train_lda(dictionary_1, corpus_1, topics)

            os.makedirs(os.path.dirname(f"{data_folder}lda-gpt2_nt-gpt2_nt/gpt2_nt_1{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                        exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-gpt2_nt/gpt2_nt_1{extra_folder}/{combi}/{topics}/corpus_{topics}", "w") as file:
                json.dump(corpus_1, file, indent=2)
            dictionary_1.save(f"{data_folder}lda-gpt2_nt-gpt2_nt/gpt2_nt_1{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_1.save(f"{data_folder}lda-gpt2_nt-gpt2_nt/gpt2_nt_1{extra_folder}/{combi}/{topics}/ldamodel_{topics}")
        else:
            model_2 = train_lda(dictionary_2, corpus_2, topics)

            os.makedirs(os.path.dirname(f"{data_folder}lda-gpt2_nt-gpt2_nt/gpt2_nt_2{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                        exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-gpt2_nt/gpt2_nt_2{extra_folder}/{combi}/{topics}/corpus_{topics}", "w") as file:
                json.dump(corpus_2, file, indent=2)
            dictionary_2.save(f"{data_folder}lda-gpt2_nt-gpt2_nt/gpt2_nt_2{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_2.save(f"{data_folder}lda-gpt2_nt-gpt2_nt/gpt2_nt_2{extra_folder}/{combi}/{topics}/ldamodel_{topics}")

    # Create LDA Model for gpt2 and gpt2
    elif dataset == 5 or dataset == 6:
        data_train_1 = load_json(f"{data_folder}dataset1-gpt2.json")
        data_train_2 = load_json(f"{data_folder}dataset2-gpt2.json")
        dictionary_1, corpus_1, dictionary_2, corpus_2 = tokenize_special(data_train_1, data_train_2, union)

        set_seed(seed2)
        if dataset == 5:
            model_1 = train_lda(dictionary_1, corpus_1, topics)

            os.makedirs(os.path.dirname(f"{data_folder}lda-gpt2-gpt2/gpt2_1{extra_folder}/{combi}/{topics}/corpus_{topics}"), exist_ok=True)
            with open(f"{data_folder}lda-gpt2-gpt2/gpt2_1{extra_folder}/{combi}/{topics}/corpus_{topics}", "w") as file:
                json.dump(corpus_1, file, indent=2)
            dictionary_1.save(f"{data_folder}lda-gpt2-gpt2/gpt2_1{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_1.save(f"{data_folder}lda-gpt2-gpt2/gpt2_1{extra_folder}/{combi}/{topics}/ldamodel_{topics}")
        else:
            model_2 = train_lda(dictionary_2, corpus_2, topics)

            os.makedirs(os.path.dirname(f"{data_folder}lda-gpt2-gpt2/gpt2_2{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                        exist_ok=True)
            with open(f"{data_folder}lda-gpt2-gpt2/gpt2_2{extra_folder}/{combi}/{topics}/corpus_{topics}", "w") as file:
                json.dump(corpus_2, file, indent=2)
            dictionary_2.save(f"{data_folder}lda-gpt2-gpt2/gpt2_2{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_2.save(f"{data_folder}lda-gpt2-gpt2/gpt2_2{extra_folder}/{combi}/{topics}/ldamodel_{topics}")

    # Create LDA Model for wiki_nt and arxiv
    elif dataset == 7 or dataset == 8:
        data_train_1 = load_wikitext(samples)
        data_train_2 = load_arxiv(samples)
        dictionary_1, corpus_1, dictionary_2, corpus_2 = tokenize_special(data_train_1, data_train_2, union)

        set_seed(seed2)
        if dataset == 7:
            model_1 = train_lda(dictionary_1, corpus_1, topics)

            os.makedirs(os.path.dirname(f"{data_folder}lda-wiki_nt-arxiv/wiki_nt{extra_folder}/{combi}/{topics}/corpus_{topics}"), exist_ok=True)
            with open(f"{data_folder}lda-wiki_nt-arxiv/wiki_nt{extra_folder}/{combi}/{topics}/corpus_{topics}", "w") as file:
                json.dump(corpus_1, file, indent=2)
            dictionary_1.save(f"{data_folder}lda-wiki_nt-arxiv/wiki_nt{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_1.save(f"{data_folder}lda-wiki_nt-arxiv/wiki_nt{extra_folder}/{combi}/{topics}/ldamodel_{topics}")
        else:
            model_2 = train_lda(dictionary_2, corpus_2, topics)

            os.makedirs(os.path.dirname(f"{data_folder}lda-wiki_nt-arxiv/arxiv{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                        exist_ok=True)
            with open(f"{data_folder}lda-wiki_nt-arxiv/arxiv{extra_folder}/{combi}/{topics}/corpus_{topics}", "w") as file:
                json.dump(corpus_2, file, indent=2)
            dictionary_2.save(f"{data_folder}lda-wiki_nt-arxiv/arxiv{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_2.save(f"{data_folder}lda-wiki_nt-arxiv/arxiv{extra_folder}/{combi}/{topics}/ldamodel_{topics}")

    # Create LDA Model for arxiv and gpt2_nt
    elif dataset == 9 or dataset == 10:
        data_train_1 = load_json(f"{data_folder}dataset1-gpt2-wiki_nt.json")
        data_train_2 = load_arxiv(samples)
        dictionary_1, corpus_1, dictionary_2, corpus_2 = tokenize_special(data_train_1, data_train_2, union)

        set_seed(seed2)
        if dataset == 9:
            model_1 = train_lda(dictionary_1, corpus_1, topics)

            os.makedirs(os.path.dirname(f"{data_folder}lda-gpt2_nt-arxiv/gpt2_nt{extra_folder}/{combi}/{topics}/corpus_{topics}"), exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-arxiv/gpt2_nt{extra_folder}/{combi}/{topics}/corpus_{topics}", "w") as file:
                json.dump(corpus_1, file, indent=2)
            dictionary_1.save(f"{data_folder}lda-gpt2_nt-arxiv/gpt2_nt{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_1.save(f"{data_folder}lda-gpt2_nt-arxiv/gpt2_nt{extra_folder}/{combi}/{topics}/ldamodel_{topics}")
        else:
            model_2 = train_lda(dictionary_2, corpus_2, topics)

            os.makedirs(os.path.dirname(f"{data_folder}lda-gpt2_nt-arxiv/arxiv{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                        exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-arxiv/arxiv{extra_folder}/{combi}/{topics}/corpus_{topics}", "w") as file:
                json.dump(corpus_2, file, indent=2)
            dictionary_2.save(f"{data_folder}lda-gpt2_nt-arxiv/arxiv{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_2.save(f"{data_folder}lda-gpt2_nt-arxiv/arxiv{extra_folder}/{combi}/{topics}/ldamodel_{topics}")

    # Create LDA Model for arxiv and gpt2_nt-top_p
    elif dataset == 11 or dataset == 12:
        data_train_1 = load_json(f"{data_folder}dataset1-gpt2-wiki_nt-top_p.json")
        data_train_2 = load_arxiv(samples)
        dictionary_1, corpus_1, dictionary_2, corpus_2 = tokenize_special(data_train_1, data_train_2, union)

        set_seed(seed2)
        if dataset == 11:
            model_1 = train_lda(dictionary_1, corpus_1, topics)

            os.makedirs(os.path.dirname(
                f"{data_folder}lda-gpt2_nt-arxiv-top_p/gpt2_nt{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                        exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-arxiv-top_p/gpt2_nt{extra_folder}/{combi}/{topics}/corpus_{topics}",
                      "w") as file:
                json.dump(corpus_1, file, indent=2)
            dictionary_1.save(
                f"{data_folder}lda-gpt2_nt-arxiv-top_p/gpt2_nt{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_1.save(f"{data_folder}lda-gpt2_nt-arxiv-top_p/gpt2_nt{extra_folder}/{combi}/{topics}/ldamodel_{topics}")
        else:
            model_2 = train_lda(dictionary_2, corpus_2, topics)

            os.makedirs(
                os.path.dirname(f"{data_folder}lda-gpt2_nt-arxiv-top_p/arxiv{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-arxiv-top_p/arxiv{extra_folder}/{combi}/{topics}/corpus_{topics}",
                      "w") as file:
                json.dump(corpus_2, file, indent=2)
            dictionary_2.save(
                f"{data_folder}lda-gpt2_nt-arxiv-top_p/arxiv{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_2.save(f"{data_folder}lda-gpt2_nt-arxiv-top_p/arxiv{extra_folder}/{combi}/{topics}/ldamodel_{topics}")

    # Create LDA Model for wikitext and gpt2_nt-top_p
    elif dataset == 13 or dataset == 14:
        data_train_1 = load_json(f"{data_folder}dataset1-gpt2-wiki_nt-top_p.json")
        data_train_2 = load_wikitext(samples)
        dictionary_1, corpus_1, dictionary_2, corpus_2 = tokenize_special(data_train_1, data_train_2, union)

        set_seed(seed2)
        if dataset == 13:
            model_1 = train_lda(dictionary_1, corpus_1, topics)

            os.makedirs(os.path.dirname(
                f"{data_folder}lda-gpt2_nt-wiki_nt-top_p/gpt2_nt{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                        exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-wiki_nt-top_p/gpt2_nt{extra_folder}/{combi}/{topics}/corpus_{topics}",
                      "w") as file:
                json.dump(corpus_1, file, indent=2)
            dictionary_1.save(
                f"{data_folder}lda-gpt2_nt-wiki_nt-top_p/gpt2_nt{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_1.save(f"{data_folder}lda-gpt2_nt-wiki_nt-top_p/gpt2_nt{extra_folder}/{combi}/{topics}/ldamodel_{topics}")
        else:
            model_2 = train_lda(dictionary_2, corpus_2, topics)

            os.makedirs(
                os.path.dirname(f"{data_folder}lda-gpt2_nt-wiki_nt-top_p/wiki_nt{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-wiki_nt-top_p/wiki_nt{extra_folder}/{combi}/{topics}/corpus_{topics}",
                      "w") as file:
                json.dump(corpus_2, file, indent=2)
            dictionary_2.save(
                f"{data_folder}lda-gpt2_nt-wiki_nt-top_p/wiki_nt{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_2.save(f"{data_folder}lda-gpt2_nt-wiki_nt-top_p/wiki_nt{extra_folder}/{combi}/{topics}/ldamodel_{topics}")

    # Create LDA Model for gpt2_nt-top_p and gpt2_nt-top_p
    elif dataset == 15 or dataset == 16:
        data_train_1 = load_json(f"{data_folder}dataset1-gpt2-wiki_nt-top_p.json")
        data_train_2 = load_json(f"{data_folder}dataset2-gpt2-wiki_nt-top_p.json")
        dictionary_1, corpus_1, dictionary_2, corpus_2 = tokenize_special(data_train_1, data_train_2, union)

        set_seed(seed2)
        if dataset == 15:
            model_1 = train_lda(dictionary_1, corpus_1, topics)

            os.makedirs(os.path.dirname(f"{data_folder}lda-gpt2_nt-gpt2_nt-top_p/gpt2_nt_1{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                        exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-gpt2_nt-top_p/gpt2_nt_1{extra_folder}/{combi}/{topics}/corpus_{topics}", "w") as file:
                json.dump(corpus_1, file, indent=2)
            dictionary_1.save(f"{data_folder}lda-gpt2_nt-gpt2_nt-top_p/gpt2_nt_1{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_1.save(f"{data_folder}lda-gpt2_nt-gpt2_nt-top_p/gpt2_nt_1{extra_folder}/{combi}/{topics}/ldamodel_{topics}")
        else:
            model_2 = train_lda(dictionary_2, corpus_2, topics)

            os.makedirs(os.path.dirname(f"{data_folder}lda-gpt2_nt-gpt2_nt-top_p/gpt2_nt_2{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                        exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-gpt2_nt-top_p/gpt2_nt_2{extra_folder}/{combi}/{topics}/corpus_{topics}", "w") as file:
                json.dump(corpus_2, file, indent=2)
            dictionary_2.save(f"{data_folder}lda-gpt2_nt-gpt2_nt-top_p/gpt2_nt_2{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_2.save(f"{data_folder}lda-gpt2_nt-gpt2_nt-top_p/gpt2_nt_2{extra_folder}/{combi}/{topics}/ldamodel_{topics}")
    ##
    # Create LDA Model for arxiv and gpt2_nt-typ_p
    elif dataset == 17 or dataset == 18:
        data_train_1 = load_json(f"{data_folder}dataset1-gpt2-wiki_nt-typ_p.json")
        data_train_2 = load_arxiv(samples)
        dictionary_1, corpus_1, dictionary_2, corpus_2 = tokenize_special(data_train_1, data_train_2, union)

        set_seed(seed2)
        if dataset == 17:
            model_1 = train_lda(dictionary_1, corpus_1, topics)

            os.makedirs(os.path.dirname(
                f"{data_folder}lda-gpt2_nt-arxiv-typ_p/gpt2_nt{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-arxiv-typ_p/gpt2_nt{extra_folder}/{combi}/{topics}/corpus_{topics}",
                      "w") as file:
                json.dump(corpus_1, file, indent=2)
            dictionary_1.save(
                f"{data_folder}lda-gpt2_nt-arxiv-typ_p/gpt2_nt{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_1.save(
                f"{data_folder}lda-gpt2_nt-arxiv-typ_p/gpt2_nt{extra_folder}/{combi}/{topics}/ldamodel_{topics}")
        else:
            model_2 = train_lda(dictionary_2, corpus_2, topics)

            os.makedirs(
                os.path.dirname(
                    f"{data_folder}lda-gpt2_nt-arxiv-typ_p/arxiv{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-arxiv-typ_p/arxiv{extra_folder}/{combi}/{topics}/corpus_{topics}",
                      "w") as file:
                json.dump(corpus_2, file, indent=2)
            dictionary_2.save(
                f"{data_folder}lda-gpt2_nt-arxiv-typ_p/arxiv{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_2.save(f"{data_folder}lda-gpt2_nt-arxiv-typ_p/arxiv{extra_folder}/{combi}/{topics}/ldamodel_{topics}")

    # Create LDA Model for wikitext and gpt2_nt-typ_p
    elif dataset == 19 or dataset == 20:
        data_train_1 = load_json(f"{data_folder}dataset1-gpt2-wiki_nt-typ_p.json")
        data_train_2 = load_wikitext(samples)
        dictionary_1, corpus_1, dictionary_2, corpus_2 = tokenize_special(data_train_1, data_train_2, union)

        set_seed(seed2)
        if dataset == 19:
            model_1 = train_lda(dictionary_1, corpus_1, topics)

            os.makedirs(os.path.dirname(
                f"{data_folder}lda-gpt2_nt-wiki_nt-typ_p/gpt2_nt{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-wiki_nt-typ_p/gpt2_nt{extra_folder}/{combi}/{topics}/corpus_{topics}",
                      "w") as file:
                json.dump(corpus_1, file, indent=2)
            dictionary_1.save(
                f"{data_folder}lda-gpt2_nt-wiki_nt-typ_p/gpt2_nt{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_1.save(
                f"{data_folder}lda-gpt2_nt-wiki_nt-typ_p/gpt2_nt{extra_folder}/{combi}/{topics}/ldamodel_{topics}")
        else:
            model_2 = train_lda(dictionary_2, corpus_2, topics)

            os.makedirs(
                os.path.dirname(
                    f"{data_folder}lda-gpt2_nt-wiki_nt-typ_p/wiki_nt{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-wiki_nt-typ_p/wiki_nt{extra_folder}/{combi}/{topics}/corpus_{topics}",
                      "w") as file:
                json.dump(corpus_2, file, indent=2)
            dictionary_2.save(
                f"{data_folder}lda-gpt2_nt-wiki_nt-typ_p/wiki_nt{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_2.save(
                f"{data_folder}lda-gpt2_nt-wiki_nt-typ_p/wiki_nt{extra_folder}/{combi}/{topics}/ldamodel_{topics}")

    # Create LDA Model for gpt2_nt-typ_p and gpt2_nt-typ_p
    elif dataset == 21 or dataset == 22:
        data_train_1 = load_json(f"{data_folder}dataset1-gpt2-wiki_nt-typ_p.json")
        data_train_2 = load_json(f"{data_folder}dataset2-gpt2-wiki_nt-typ_p.json")
        dictionary_1, corpus_1, dictionary_2, corpus_2 = tokenize_special(data_train_1, data_train_2, union)

        set_seed(seed2)
        if dataset == 21:
            model_1 = train_lda(dictionary_1, corpus_1, topics)

            os.makedirs(
                os.path.dirname(f"{data_folder}lda-gpt2_nt-gpt2_nt-typ_p/gpt2_nt_1{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-gpt2_nt-typ_p/gpt2_nt_1{extra_folder}/{combi}/{topics}/corpus_{topics}",
                      "w") as file:
                json.dump(corpus_1, file, indent=2)
            dictionary_1.save(f"{data_folder}lda-gpt2_nt-gpt2_nt-typ_p/gpt2_nt_1{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_1.save(f"{data_folder}lda-gpt2_nt-gpt2_nt-typ_p/gpt2_nt_1{extra_folder}/{combi}/{topics}/ldamodel_{topics}")
        else:
            model_2 = train_lda(dictionary_2, corpus_2, topics)

            os.makedirs(
                os.path.dirname(f"{data_folder}lda-gpt2_nt-gpt2_nt-typ_p/gpt2_nt_2{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-gpt2_nt-typ_p/gpt2_nt_2{extra_folder}/{combi}/{topics}/corpus_{topics}",
                      "w") as file:
                json.dump(corpus_2, file, indent=2)
            dictionary_2.save(f"{data_folder}lda-gpt2_nt-gpt2_nt-typ_p/gpt2_nt_2{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_2.save(f"{data_folder}lda-gpt2_nt-gpt2_nt-typ_p/gpt2_nt_2{extra_folder}/{combi}/{topics}/ldamodel_{topics}")


main()
