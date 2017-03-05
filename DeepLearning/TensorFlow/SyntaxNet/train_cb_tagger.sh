#!/usr/bin/env bash
docker run -it -v /Users/simon.hughes/data/tensorflow/syntaxnet/tagger:/opt/tagger syntaxnet:works  /bin/bash
# then:
# cd /opt/tagger/cb
# sh train_tagger_docker.sh