var collectionNames = db.getCollectionNames(), stats = [];
collectionNames.forEach(function (n) { stats.push(db[n].stats()); });
stats = stats.sort(function(a, b) { return b['size'] - a['size']; });
for (var c in stats) { 
    print(stats[c]['ns'] + ": " + stats[c]['size']/1000000 + "M (" + stats[c]['storageSize']/1000000 + "M)"); 
}