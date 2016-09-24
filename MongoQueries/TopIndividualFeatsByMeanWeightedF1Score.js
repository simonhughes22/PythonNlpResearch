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
{
    $match:{
        count: {  $eq:1 },
        window_size: { $eq:9 }
    }
},
{    $unwind: "$feats"},
{
    $group: { 
        _id:{ weighted_f1_score: "$weighted_f1_score", window_size: "$window_size", feats: "$feats"} 
    }
},
{
    $project: { 
        weighted_f1_score: "$_id.weighted_f1_score", window_size: "$_id.window_size", feats: "$_id.feats",
       _id: 0 
    }
},
{
    $sort:{
        "weighted_f1_score":-1,
        //asof: -1
        //count: -1
    }
},
])