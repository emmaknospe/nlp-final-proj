from load import Document


def display_coreferences(doc, coreferences):
    assert isinstance(doc, Document)
    for i, sentence in enumerate(doc.sentences):
        for j, coreference in enumerate(coreferences):
            for mention in coreference:
                if mention[0] == i:
                    for k in range(mention[1], mention[2]):
                        sentence[k] = sentence[k] + "(" + str(j) + ")"
        print(" ".join(sentence))