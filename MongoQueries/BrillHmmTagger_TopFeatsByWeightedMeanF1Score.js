db.getCollection('CB_TAGGING_VD_BrillTagger_HMM').aggregate(
[{
    $project: { 
            weighted_f1_score: "$WEIGHTED_MEAN_CONCEPT_CODES.f1_score",
            window_size: "$parameters.window_size",
            feats: "$parameters.extractors",
            minscore: "$parameters.MIN_SCORE",
            maxrules: "$parameters.MAX_RULES",
            base_tagger: "$parameters.BASE_TAGGER",
            count: { $size:"$parameters.extractors" },
            //asof: 0,
            "_id":0
    }
},
{
    $match:{
        //count: {  $eq:2 },
        //window_size: { $eq:9 }
    }
},
{    $sort:{
        //count: 1,
        "weighted_f1_score":-1,
        //asof: -1
        
    }
},
])