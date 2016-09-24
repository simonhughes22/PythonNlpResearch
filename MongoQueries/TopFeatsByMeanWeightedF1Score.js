db.getCollection('CB_TAGGING_VD_FEAT_SELECTION').aggregate(
[{
    $project: { 
            weighted_f1_score: "$WEIGHTED_MEAN_CONCEPT_CODES.f1_score",
            window_size: "$parameters.window_size",
            feats: "$parameters.extractors",
            count: { $size:"$parameters.extractors" },
            //asof: 0,
            "_id":0
    }
},
{    $sort:{
        //count: 1,
        "weighted_f1_score":-1,
        //asof: -1
        
    }
},
])