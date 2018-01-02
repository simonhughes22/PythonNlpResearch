#!/usr/bin/env bash
# --smallfiles causes deafult file size to drop to 64M and quotafiles limits to 1 file (might want to up that)
#   see https://stackoverflow.com/questions/9779923/set-mongodb-database-quota-size
# to shrink the db, launch with these options then use mongodump and mongorestore to export / import the db
#   mongodump -d metrics_causal -o MongoExport
#   mongorestore -d metrics_causal_bak  MongoExport/metrics_causal/
mongod -dbpath=/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/MongoDb/ --smallfiles --quotaFiles 8
