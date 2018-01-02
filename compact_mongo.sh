#!/usr/bin/env bash
# launch mongod with small files and run a repair db (--repair). Will need to restart it once more after completed
mongod -dbpath=/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/MongoDb/ --smallfiles --quotaFiles 1 --repair