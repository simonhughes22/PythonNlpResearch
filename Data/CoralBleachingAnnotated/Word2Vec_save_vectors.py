from load_data import load_process_essays

from window_based_tagger_config import get_config
from Decorators import memoize_to_disk
import Settings
from Word2Vec_to_file import vectors_to_pickled_dict

settings = Settings.Settings()

#folder =                            settings.data_directory + "CoralBleaching/BrattData/EBA_Pre_Post_Merged/"
folder =                            settings.data_directory + "CoralBleaching/BrattData/EBA1415_Merged/"
processed_essay_filename_prefix =   settings.data_directory + "CoralBleaching/BrattData/Pickled/essays_proc_pickled_"

config = get_config(folder)
config["min_df"] = 1
config["remove_infrequent"] = False
config["include_vague"] = True
config["include_normal"] = True
config["stem"] = False
config["remove_stop_words"] = False
config["remove_punctuation"] = True
# Note: See below. I store the regular and lc versions of each word
config["lower_case"] = False

""" FEATURE EXTRACTION """
""" LOAD DATA """
print("Loading Essays")
mem_process_essays = memoize_to_disk(filename_prefix=processed_essay_filename_prefix)(load_process_essays)
tagged_essays = mem_process_essays( **config )

words = set()
for essay in tagged_essays:
    for sentence in essay.sentences:
        for word, tags in sentence:
            # store common variants of each word
            words.add(word)
            words.add(word.lower())
            words.add(word[0].upper() + word[1:].lower())

words.add("a")
words.add("an")
words.add("the")
words.add("then")
words.add("he")
words.add("she")
words.add("it")
words.add("them")
words.add("that")
words.add("which")

print("%i words loaded" % len(words))
print("Saving Data")
vectors_to_pickled_dict(words, settings.data_directory + "CoralBleaching/Word2Vec_Vectors.pl")
print("Done")
