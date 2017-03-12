#!/usr/bin/env bash

# installed from source
#source activate theano_py35_bleedingedge
#source activate theano_py35

# both from source
source activate keras_and_theano_bleeding_edge

export PYTHONPATH='/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/PythonNlpResearch:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Calculations:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Classifiers:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Classifiers/RegEx:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Classifiers/Trees:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Clustering:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/CodeGen:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/CollectionsHelper:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Data:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Data/Chicago:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Data/CoralBleaching:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Data/CoralBleachingWordTagger:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Data/SkinCancerWordTagger:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Data/GlobalWarming:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Data/GlobalWarmingAnnotated:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Data/GoogleNGrams:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/DeepLearning:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/DeepLearning/DeepBeliefNetwork:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/DeepLearning/Examples:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/DeepLearning/Examples/DeepAutoEncoder:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/DeepLearning/Examples/DeepAutoEncoder_MichaelNielsen:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/DeepLearning/MyCode:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/DeepLearning/MyCode/RecursiveAutoEncoder:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/DeepLearning/WordVectors:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/Chicago:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/CoralBleaching:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/GlobalWarming:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/GlobalWarming/Causal:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/GlobalWarming/WordClustering:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/FeatureExtraction/Text:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Frequency:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/LanguageModel:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Results:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Tagging:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/VectorSpace:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/VectorSpace/WordVectors:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/VectorSpace/WordVectors/Word2Vec:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/_Scratch/Sub:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Synonyms:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/DeepLearning/MyCode/:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/MontyLingua:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/CoralBleachingWordTagger/WindowBasedClassifier:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/Tagging:/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/CoralBleachingWordTagger/DeepNN'

export DYLD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME:$CUDA_HOME/extras/CUPTI/lib"
export LD_LIBRARY_PATH=$DYLD_LIBRARY_PATH

# see ~/.theanorc for defaults (and to override defaults)
# ENV VARS below override defaults
export THEANO_FLAGS='mode=FAST_RUN,device=gpu,floatX=float32'

# Need to override BACKEND to be THEANO
# see ~/.keras/keras.json for default settings
export KERAS_BACKEND='theano'
# I think this sets it to THEANO ordering - https://keras.io/backend/
export KERAS_IMAGE_DIM_ORDERING='th'

cd Notebooks
jupyter notebook
