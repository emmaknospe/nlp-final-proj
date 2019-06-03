from load import Document


def model_all_mentions_paired(doc):
    return [list(doc.mentions)]

def model_all_mentions_unpaired(doc):
    generated_coreferences = []
    for mention in doc.mentions:
        generated_coreferences.append([mention])
    return generated_coreferences
