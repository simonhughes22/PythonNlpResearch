db.getCollection('CB_TAGGING_VD_FEAT_SELECTION').aggregate(
[{
    $project: { 
            weighted_f1_score: "$WEIGHTED_MEAN_CONCEPT_CODES.f1_score",
            macro_f1: "$MACRO_F1",
            window_size: "$parameters.window_size",
            feats: "$parameters.extractors",
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
        //"macro_f1":-1,
        "weighted_f1_score":-1,
        //asof: -1
        
    }
},
])