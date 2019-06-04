from collections import defaultdict
from load import Document
from sklearn.linear_model import LogisticRegression
import tqdm


def get_mention_as_str(doc, mention):
    return ' '.join(doc.sentences[mention[0]][mention[1]:mention[2]])

def get_mention_as_list(doc, mention):
    return doc.sentences[mention[0]][mention[1]:mention[2]]


def either_mention_contains_any_of(mention1_list, mention2_list, tokens):
    for token in tokens:
        if token in mention1_list or token in mention2_list:
            return True
    return False


def get_feature_vector(mention1, mention2, doc):
    vector = [0 for _ in range(7)]
    mention1_str = get_mention_as_str(doc, mention1)
    mention2_str = get_mention_as_str(doc, mention2)
    mention1_list = get_mention_as_list(doc, mention1)
    mention2_list = get_mention_as_list(doc, mention2)
    vector[0] = 1 if mention1_str == mention2_str else 0
    third_person_male_pronouns = ['he', 'him', 'his']
    third_person_female_pronouns = ['she', 'her', 'hers']
    second_person_pronouns = ['you', 'yours']
    first_person_pronouns = ['i', 'me', 'my']
    third_person_neutral_pronouns = ['they', 'them', 'their', 'theirs']
    if mention1_str.lower() in third_person_male_pronouns and mention2_str.lower() in third_person_male_pronouns:
        vector[1] = 1
    else:
        vector[1] = 0

    if mention1_str.lower() in third_person_female_pronouns and mention2_str.lower() in third_person_female_pronouns:
        vector[2] = 1
    else:
        vector[2] = 0

    if mention1_str.lower() in second_person_pronouns and mention2_str.lower() in second_person_pronouns:
        vector[3] = 1
    else:
        vector[3] = 0

    if mention1_str.lower() in first_person_pronouns and mention2_str.lower() in first_person_pronouns:
        vector[4] = 1
    else:
        vector[4] = 0

    if mention1_str.lower() in third_person_neutral_pronouns and mention2_str.lower() in third_person_neutral_pronouns:
        vector[5] = 1
    else:
        vector[5] = 0
    vector[6] = len(set(mention1_list).intersection(set(mention2_list)))
    return vector


def get_lr_model(docs, verbose=True):
    vectors = []
    target = []
    if verbose:
        iterator = tqdm.tqdm(docs)
    else:
        iterator = docs
    for doc in iterator:
        assert isinstance(doc, Document)
        values = defaultdict(lambda: defaultdict(bool))
        for cluster in doc.real_coreferences:
            for mention1 in cluster:
                for mention2 in cluster:
                    values[tuple(mention1)][tuple(mention2)] = True
        for mention1 in doc.mentions:
            for mention2 in doc.mentions:
                vectors.append(get_feature_vector(mention1, mention2, doc))
                target.append(values[tuple(mention1)][tuple(mention2)])

    if verbose:
        print("Fitting model...")
    model = LogisticRegression(solver="liblinear")
    model.fit(vectors, target)
    if verbose:
        print("Model fitted.")
    return model


class LinearEdgePredictor:
    def __init__(self, docs, verbose=True):
        self.lr_model = get_lr_model(docs, verbose=verbose)

    def should_be_joined(self, mention1, mention2, doc):
        vector = get_feature_vector(mention1, mention2, doc)
        return self.lr_model.predict([vector])[0]

    def get_predictions(self, doc):
        clusters = []
        added = set()
        for mention1 in doc.mentions:
            if mention1 not in added:
                cluster = [mention1]
                added.add(mention1)
                for mention2 in doc.mentions:
                    if mention2 not in added and self.should_be_joined(mention1, mention2, doc):
                        cluster.append(mention2)
                        added.add(mention2)
                clusters.append(cluster)
        return clusters