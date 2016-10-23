db.getCollection('CB_TAGGING_VD')
.aggregate(
[{
    $project: { 
            weighted_f1_score: "$WEIGHTED_MEAN_CONCEPT_CODES.f1_score",
            //macro_f1: "$MACRO_F1",
            micro_f1_score: "$MICRO_F1.f1_score",
            window_size: "$parameters.window_size",
            num_iterations: "$parameters.num_iterations",
            tag_history: "$parameters.tag_history",
            feats: "$parameters.extractors",
            count: { $size:"$parameters.extractors" },
            algorithm : 1,
            //asof: 0,
            "_id":0
    }
}
,
{
    $match:{
        //count: {  $eq:2 },
        //window_size: { $eq:9 }
        algorithm: "AveragedPerceptronBinary"
        //micro_f1_score: { $exists : true }
    }
},
{    $sort:{
        //count: 1,
        //"macro_f1":-1,
        "weighted_f1_score":-1,
        //"micro_f1_score":-1,
        //asof: -1
        
    }
},
])