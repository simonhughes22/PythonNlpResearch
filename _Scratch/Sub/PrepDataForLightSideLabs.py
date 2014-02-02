__author__ = 'simon.hughes'

import GwData
import WordTokenizer
import Settings

from GwExperimentBase import important_codes

settings = Settings.Settings()

data = GwData.GwData()

tokenized_docs = WordTokenizer.tokenize(data.documents, min_word_count=5, stem=False, lemmatize=False)

folder = settings.data_directory + "GlobalWarming\\LightSideLabs\\"
for code in important_codes:
    lbls = data.labels_for(code)

    fname = folder + code + ".csv"
    with open(fname, "w+") as f:
        f.write("id,class,content\n")
        id = 0
        for lbl, sentence in zip(lbls, tokenized_docs):
            if len(sentence) == 0:
                continue

            str_sentence = " ".join(sentence)
            f.write(str(id) + "," +  str(lbl) + "," + str_sentence.strip() + "\n")
            id += 1