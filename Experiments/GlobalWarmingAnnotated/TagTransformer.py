# coding=utf-8
# Collapse tag variants
def __tag_transformer__(tag):
    # Collapse PXX to XX
    #if tag[0] == "P":
    #    return tag[1:]
    #if ":P" in tag:
    #    tag = tag.replace(":P", ":")
    # Collapse XX.Y to XX
    if "." in tag:
        tag = tag[:tag.index(".")]
    # Collapse XX_Y to XX
    if "_" in tag:
        tag = tag[:tag.index("_")]
    return tag


def transform_essay_tags(tagged_essays):
    for essay in tagged_essays:
        transformed_sentences = []
        for tagged_sentence in essay.sentences:
            t_sentence = [(wd, map(__tag_transformer__, tags)) for (wd, tags) in tagged_sentence]
            transformed_sentences.append(t_sentence)
        essay.sentences = transformed_sentences

def replace_periods(tagged_essays):
    for essay in tagged_essays:
        transformed_sentences = []
        for tagged_sentence in essay.sentences:
            t_sentence = [(wd, map(lambda t: t.replace(".", "-"), tags)) for (wd, tags) in tagged_sentence]
            transformed_sentences.append(t_sentence)
        essay.sentences = transformed_sentences
