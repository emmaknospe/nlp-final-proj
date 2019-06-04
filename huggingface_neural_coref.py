import neuralcoref
import spacy
from spacy.tokens.doc import Doc
from load import Document


def our_custom_function(doc, hashes, blacklist=False):
    print(doc)
    print(hashes)
    print(type(doc))
    print(type(hashes))

def get_sentence_position_from_index(doc):
    position_to_sentence = {}
    index = 0
    for i, sentence in enumerate(doc.sentences):
        for j, token in enumerate(sentence):
            position_to_sentence[index] = (i, j)
            index += 1
    for j in range(100):
        position_to_sentence[index] = (len(doc.sentences) - 1, len(doc.sentences[-1]) - 1)
        index += 1
    return position_to_sentence

def run_hugging_face(data):
    nlp = spacy.load("en", disable=['tokenizer'])
    custom_token_override = get_custom_token_override(data, nlp)
    nlp.add_pipe(custom_token_override, first=True)
    neuralcoref.add_to_pipe(nlp)
    doc = nlp(u' '.join(token for sentence in data.sentences for token in sentence))
    position_to_sentence = get_sentence_position_from_index(data)
    coreferences = []
    for cluster in doc._.coref_clusters:
        cluster_list = []
        for mention in cluster.mentions:
            sentence, start = position_to_sentence[mention.start]
            sentence, end = position_to_sentence[mention.end]
            cluster_list.append((sentence, start, end))
        coreferences.append(cluster_list)
    return coreferences


def get_custom_token_override(data, nlp):
    def custom_token_override(doc):
        assert isinstance(doc, spacy.tokens.doc.Doc)
        return Doc(vocab=nlp.vocab, words=[token for sentence in data.sentences for token in sentence])
    return custom_token_override


if __name__ == "__main__":
    doc = Document([["Jack", "likes", "twitter", "."], ["It", "makes", "him", "happy", "."]], [], [])
    run_hugging_face(doc)

