#!/usr/bin/env bash
# --smallfiles causes deafult file size to drop to 64M and quotafiles limits to 1 file (might want to up that)
#   see https://stackoverflow.com/questions/9779923/set-mongodb-database-quota-size
# to shrink the db, launch with these options then use mongodump and mongorestore to export / import the db
#   mongodump -d metrics_causal -o MongoExport
#   mongorestore -d metrics_causal_bak  MongoExport/metrics_causal/

# NOTE on mongo version - These files only work on mongo 3.4. Latest mongo (3.6+, 4.0+) won't read them
#  The SECOND answer here explains how to install the older version https://stackoverflow.com/questions/30379127/how-to-install-earlier-version-of-mongodb-with-homebrew

#mongod -dbpath=/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/MongoDb/ --smallfiles --quotaFiles 8
mongod -dbpath=/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/MongoDBStatSig/ --smallfiles --quotaFiles 8
