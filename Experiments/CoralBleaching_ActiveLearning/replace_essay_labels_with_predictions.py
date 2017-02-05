def replace_essay_labels_with_predictions(essay_feats, word_feats, tag2Classifier, confidence_threshold=None):

    if not confidence_threshold:
        cls = list(tag2Classifier.values())[0]
        if hasattr(cls, "decision_function"):
            confidence_threshold = 0.0
        else: # assume a probability
            confidence_threshold = 0.5

    # dicts, key = tag, to a 1D array of word-level predictions
    real_num_predictions_bytag = dict()
    for tag in tag2Classifier.keys():

        cls = tag2Classifier[tag]
        if hasattr(cls, "decision_function"):
            real_num_predictions = cls.decision_function(word_feats)
        else:
            real_num_predictions = cls.predict_proba(word_feats)
        real_num_predictions_bytag[tag] = real_num_predictions

    # filter to the expected tags
    valid_tags = set(tag2Classifier.keys())

    # build a new list of processessays.Essay objects, each of which contains a list of sentences
    # each of which is a list of featureextractortransformer.Word objects, which contain
    # a word, it's associated tags (which we are swapping out) and it's features
    new_essay_feats = []
    for essay_ix, essay in enumerate(essay_feats):

        new_sentences = []
        # Essay is from ./Data/processessays.py
        new_essay = Essay(essay.name, sentences=new_sentences)

        new_essay_feats.append(new_essay)

        # The essay class is used in 2 different variants. Here, the
        # sentences are lists of Word objects
        for sent_ix, taggged_sentence in enumerate(essay.sentences):

            new_sentence = []
            new_sentences.append(new_sentence)
            for word_ix, (wd) in enumerate(taggged_sentence):

                predicted_tags = set()
                for tag in valid_tags:
                    real_pred = real_num_predictions_bytag[tag][word_ix]
                    if real_pred >= confidence_threshold:
                        predicted_tags.add(tag)

                # ./Data/featureextractortransformer.Word
                new_word = Word(wd.word, predicted_tags)
                # make sure to copy over the features
                new_word.features = dict(wd.features)
                # leave the vector as None (this isn't used I don't think)
                new_sentence.append(new_word)

    return new_essay_feats