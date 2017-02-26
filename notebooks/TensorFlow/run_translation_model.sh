#NOTE - need to SOURCE this file, running with sh won't work
# only works with ipython for some reason
. activate tensorflow_gpu
ipython /Users/simon.hughes/GitHub/tensorflow_models/tutorials/rnn/translate/translate.py -- --data_dir "/Users/simon.hughes/data/tensorflow/translate/cb/data_dir" --train_dir "/Users/simon.hughes/data/tensorflow/translate/cb/train_dir" --decode --size=64 --from_vocab_size=4300 --to_vocab_size=14 --num_layers=1