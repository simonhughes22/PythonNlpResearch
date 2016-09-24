db.getCollection('CB_TAGGING_VD_FEAT_SELECTION').aggregate(
[{
    $project: { 
            weighted_f1_score: "$WEIGHTED_MEAN_CONCEPT_CODES.f1_score",
            window_size: "$parameters.window_size",
            count: { $size:"$parameters.extractors" },
            asof: 1,
            "_id":0
    }
},
{
    $match:{
        //count: {  $eq:5 }
        //window_size: { $eq:9 }
    }
},
{
    $sort:{
        weighted_f1_score:-1,
        asof: -1
        //count: -1
    }
}
])