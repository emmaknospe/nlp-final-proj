from load import Document


def model_baseline(data):
    generated_coreferences = []
    for doc in data:
        assert isinstance(doc, Document)
        generated_coreferences.append([list(doc.mentions)])
    return generated_coreferences


