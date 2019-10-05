#!/usr/bin/env bash
source activate work

export PYTHONPATH='/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/API:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Calculations:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Classifiers:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Classifiers/Online:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Classifiers/RegEx:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Classifiers/StructuredLearning/SEARN:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Classifiers/Trees:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Clustering:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/CodeGen:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/CollectionsHelper:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Data:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Data/Chicago:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Data/CoralBleaching:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Data/CoralBleachingAnnotated:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Data/GlobalWarming:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Data/GlobalWarmingAnnotated:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Data/GoogleNGrams:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/DeepLearning:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/DeepLearning/DeepBeliefNetwork:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/DeepLearning/Examples:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/DeepLearning/Examples/DeepAutoEncoder:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/DeepLearning/Examples/DeepAutoEncoder_MichaelNielsen:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/DeepLearning/LSTM:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/DeepLearning/MyCode:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/DeepLearning/MyCode/RecursiveAutoEncoder:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/DeepLearning/Theano:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/DeepLearning/WordVectors:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/CoralBleachingCausalRelation/SEARN:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/CoralBleachingCausalRelation/SEARN/other_implementations:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/CoralBleachingCoRef:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/CoralBleachingWordTagger/AveragedPerceptron:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/CoralBleachingWordTagger/NLTKTaggingModels:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/CoralBleachingWordTagger/RNN_Tagger:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/CoralBleachingWordTagger/WindowBasedClassifier:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/CoralBleaching_ActiveLearning:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/SkinCancerCausalRelation/SEARN:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/SkinCancerCoRef:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/SkinCancerWordTagger:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/SkinCancerWordTagger/AveragedPerceptron:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/SkinCancerWordTagger/NLTKTaggingModels:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/SkinCancerWordTagger/RNN_Tagger:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/SkinCancerWordTagger/WindowBasedClassifier:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/SkinCancerWordTagger/WindowBasedClassifier/TaggingModels:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/Tagging:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/_Legacy/Chicago:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/_Legacy/CoralBleaching:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/_Legacy/CoralBleaching/ConvolutionalNeuralNetwork:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/_Legacy/CoralBleaching/DeepNN:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/_Legacy/CoralBleaching/FastText:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/_Legacy/CoralBleaching/MaximumEntropy:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/_Legacy/CoralBleaching/RecurrentNeuralNetwork:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/_Legacy/CoralBleaching/WindowBasedClassifier:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/_Legacy/GlobalWarming:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/_Legacy/GlobalWarming/Causal:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/_Legacy/GlobalWarming/WordClustering:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/_Legacy/GlobalWarmingAnnotated:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/_Legacy/GlobalWarmingAnnotated/AveragedPerceptron:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/_Legacy/GlobalWarmingAnnotated/WindowBasedClassifier:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/FeatureExtraction/Text:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Frequency:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/LanguageModel:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/MontyLingua:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Results:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Synonyms:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Tagging:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/VectorSpace:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/VectorSpace/WordVectors:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/VectorSpace/WordVectors/Word2Vec:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/_Scratch/Sub:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/stanford_parser'

# include pyFMexport PYTHONPATH=$PYTHONPATH:~/Software/pyFMexport MONTYLINGUA='/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/PythonNlpResearch/MontyLingua/'
# include pyFM

#export PYTHONPATH=$PYTHONPATH:~/Software/pyFM
#export MONTYLINGUA='/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/PythonNlpResearch/MontyLingua/'

#conda info --envs

cd Notebooks
jupyter lab