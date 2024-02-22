import numpy as np
import os
import gensim.downloader as api
import gensim.models as gsm
import heapq
import math
from itertools import chain
import configparser
import sys

# Check if "-dev" is passed as an argument
dev = False
if len(sys.argv) > 1 and sys.argv[1] == "-dev":
    dev = True

def dev_print(*args):
    if dev:
        __builtins__.print(*args)

if dev:
    # Read Directory from config file
    config = configparser.ConfigParser()
    config.read('.config')

    # Ensure the directory exists
    dir = config['parameter']['dir']
    if not os.path.exists(dir):
        os.makedirs(dir)
    models = api.info()["models"].keys()

    # Download the models
    dev_print("Downloading the models...")
    dev_print("Don't worry")
    dev_print("This may take a while...")
    dev_print("Actually, it will take a whole lot time...")
    dev_print("The models will be saved to", dir)
    for model in models:
        # Define the path for the model to be saved
        model_path = os.path.join(dir, model + ".model")
        # Check if the model file already exists
        if not os.path.exists(model_path):
            # Download the model only if it does not exist
            dev_print(f"Downloading {model}...")
            embedding_file = api.load(model)
            # Save the model locally
            embedding_file.save(model_path)
            dev_print(f"{model} has been downloaded and saved to {model_path}")
        else:
            dev_print(f"{model} already exists. Skipping download.")

    model_paths = [os.path.join(dir, model + ".model") for model in models]
else:
    model_paths = [
        'pretrained_word2vec_gensim/fasttext-wiki-news-subwords-300.model',
        'pretrained_word2vec_gensim/conceptnet-numberbatch-17-06-300.model',
        'pretrained_word2vec_gensim/word2vec-ruscorpora-300.model',
        'pretrained_word2vec_gensim/word2vec-google-news-300.model',
        'pretrained_word2vec_gensim/glove-wiki-gigaword-50.model',
        'pretrained_word2vec_gensim/glove-wiki-gigaword-100.model',
        'pretrained_word2vec_gensim/glove-wiki-gigaword-200.model',
        'pretrained_word2vec_gensim/glove-wiki-gigaword-300.model',
        'pretrained_word2vec_gensim/glove-twitter-25.model',
        'pretrained_word2vec_gensim/glove-twitter-50.model',
        'pretrained_word2vec_gensim/glove-twitter-100.model',
        'pretrained_word2vec_gensim/glove-twitter-200.model',
        'pretrained_word2vec_gensim/__testing_word2vec-matrix-synopsis.model'
    ]

class TrieNode:
    def __init__(self):
        self.children = {}
        self.path = None

class AppTrie:
    def __init__(self):
        self.root = TrieNode()

        # Initialize the configparser
        self.config = configparser.ConfigParser()

        # Read the config file
        self.config.read('.config')

        # Read variables from the section 'parameter'
        self.w = self.config['parameter'].getint('w')
        self.b = self.config['parameter'].getint('b')
        self.sigma = self.config['parameter'].getfloat('sigma')
        self.mu = self.config['parameter'].getfloat('mu')
        self.model_index = self.config['parameter'].getint('model_index')
        self.dir = self.config['parameter']['dir']

        # load model
        model_paths = [
            'pretrained_word2vec_gensim/fasttext-wiki-news-subwords-300.model',
            'pretrained_word2vec_gensim/conceptnet-numberbatch-17-06-300.model',
            'pretrained_word2vec_gensim/word2vec-ruscorpora-300.model',
            'pretrained_word2vec_gensim/word2vec-google-news-300.model',
            'pretrained_word2vec_gensim/glove-wiki-gigaword-50.model',
            'pretrained_word2vec_gensim/glove-wiki-gigaword-100.model',
            'pretrained_word2vec_gensim/glove-wiki-gigaword-200.model',
            'pretrained_word2vec_gensim/glove-wiki-gigaword-300.model',
            'pretrained_word2vec_gensim/glove-twitter-25.model',
            'pretrained_word2vec_gensim/glove-twitter-50.model',
            'pretrained_word2vec_gensim/glove-twitter-100.model',
            'pretrained_word2vec_gensim/glove-twitter-200.model',
            'pretrained_word2vec_gensim/__testing_word2vec-matrix-synopsis.model'
        ]
        self.model = gsm.KeyedVectors.load(model_paths[self.model_index]) 

    def insert(self, path):
        current_node = self.root
        if current_node.path is None:
            current_node.path = path.split(" > ")[0]

        increasing_path = ""
        for word in path.split(" > "):
            increasing_path = (
                increasing_path + " > " + word if increasing_path else word
            )
            if word not in current_node.children:
                current_node.children[word] = TrieNode()
            current_node.children[word].path = increasing_path
            current_node = current_node.children[word]

    def search(self, query, threshold_lower=1, threshold_upper=1.5):
        '''
        Basic A* Algorithm.
        Instead of exact match, we use two thresholds to determine the result.
        The lower threshold is the minimum distance to return the result.
        The upper threshold is the maximum distance to continue the search.
        '''

        # Sigmoid, Sim, Insert Cost and Substitute Cost exists only for Distance String
        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        def sim(word1, word2):
            if word1 == word2:
                return 1.0
            try:
                return sigmoid(self.w * self.model.similarity(word1, word2) + self.b)
            except:
                return 0

        def insert_cost(word, target, string):
            score = 0
            for i in range(len(string)):
                if string[i] == target:
                    continue
                if sim(word, string[i]) > score:
                    score = sim(word, string[i])

            return 1 - self.sigma * score + self.mu

        def substitute_cost(word, target):
            return 2 - 2 * sim(word, target)

        # Main Distance function for A* Algorithm
        # https://arxiv.org/pdf/1810.10752.pdf
        def distance_string(query, path):
            query = query.split("_")
            path = [i.split("_") for i in path.split(">")]
            path = list(chain.from_iterable(path))

            # print(query, path)
            dp = [[0 for _ in range(len(path) + 1)] for _ in range(len(query) + 1)]

            # base case
            for i in range(len(query) + 1):
                dp[i][0] = i
            for j in range(len(path) + 1):
                dp[0][j] = j

            # calculate minimum edit operations
            for i in range(1, len(query) + 1):
                for j in range(1, len(path) + 1):
                    insertion = dp[i][j - 1] + insert_cost(query[i - 1], path[j - 1], path)
                    deletion = dp[i - 1][j] + insert_cost(path[j - 1], query[i - 1], query)
                    substitution = dp[i - 1][j - 1] + substitute_cost(query[i - 1], path[j - 1])
                    dp[i][j] = min(insertion, deletion, substitution)

            return dp[len(query)][len(path)]
        
        h = []
        heapq.heappush(h, (0, 0, self.root))
        results = []

        while len(h) > 0:
            current_cost, step, current_node = heapq.heappop(h)
            a = 0.2
            b = 0.8

            for child_word, child_node in current_node.children.items():
                distance_path = distance_string(query, child_node.path)
                distance_current = distance_string(query, child_word)
                distance = a * distance_path + b * distance_current

                distance = distance_string(query, child_node.path)
                distance = distance / (step + 1)

                # print stat
                # print("Searching in path:", child_node.path, "with distance:", distance, "and step:", step + 1)

                if distance <= threshold_upper:
                    results.append(child_node.path)
                    if distance <= threshold_lower:
                        return results

                heapq.heappush(h, (distance, step + 1, child_node))

        return results
    
__all__ = ["AppTrie"]

# if main, run the test
if __name__ == "__main__":
    trie = AppTrie()
    trie.insert("select_post > new_comment > add_image")
    trie.insert("settings > battery > toggle_battery_saver")
    trie.insert("select_post > new_comment")
    trie.insert("select_post")

    test_input = [
        "turn_on_battery_saver",
        "turn_of_battery_saver",
        "comment_on_image",
        "comment_on_post",
        "see_post"
    ]

    for case in test_input:
        result = trie.search(case)
        print(result)