from load import Document
from constants import pronouns


class GraphMention:
    def __init__(self, mention, document):
        self.mention = mention
        self.document = document
        self.edges = []
        self.visited = False
    def get_token_representation(self):
        sentence_idx, start_idx, end_idx = self.mention
        sentence = self.document.sentences[sentence_idx]
        return sentence[start_idx:end_idx]


def is_pronoun(maybe_a_pronoun):
    return maybe_a_pronoun in pronouns


def exact_match(mention1_tokens, mention2_tokens, pair_pronouns=True):
    if len(mention1_tokens) == 1 and len(mention2_tokens) == 1:
        if pair_pronouns:
            return mention1_tokens[0].lower() == mention2_tokens[0].lower()
        else:
            return mention1_tokens[0].lower() == mention2_tokens[0].lower() and not is_pronoun(mention1_tokens[0])


def multiword_exact_match(mention1_tokens, mention2_tokens):
    return ' '.join(m.lower() for m in mention1_tokens) == ' '.join(m.lower() for m in mention2_tokens)


def get_component(mention):
    if mention.visited:
        return []
    else:
        elements = [mention]
        mention.visited = True
        for edge in mention.edges:
            elements.extend(get_component(edge))
        return elements


def model_pair_exact_strings(doc, match=exact_match):
    assert isinstance(doc, Document)
    mentions = [GraphMention(mention, doc) for mention in doc.mentions]
    for mention1 in mentions:
        for mention2 in mentions:
            mention1_tokens = mention1.get_token_representation()
            mention2_tokens = mention2.get_token_representation()
            if match(mention1_tokens, mention2_tokens):
                mention1.edges.append(mention2)
    clusters = []
    for mention in mentions:
        component = get_component(mention)
        if component:
            clusters.append([mention.mention for mention in component])
    return clusters


