# load.py contains functions for loading and preprocessing data
import json


def extract_mentions(doc):
    clusters = doc['mention_clusters']
    mentions = set(tuple(mention) for cluster in clusters for mention in cluster)
    return list(mentions)


def load(filename, mention_method="extract", limit=100):
    with open(filename) as file:
        docs = []
        for i, doc in enumerate(file):
            if i >= limit:
                break
            doc = json.loads(doc)
            sentences = doc['sentences']
            answers = doc['mention_clusters']
            if mention_method == "extract":
                mentions = extract_mentions(doc)
            else:
                raise Exception("Mention method {} not supported!".format(mention_method))
            doc = Document(sentences, answers, mentions)
            docs.append(doc)
        return docs


class Document:
    def __init__(self, sentences, real_coreferences, mentions):
        self.sentences = sentences
        self.real_coreferences = real_coreferences
        self.mentions = mentions


if __name__ == "__main__":
    load("PreCo_1.0/dev.jsonl")
