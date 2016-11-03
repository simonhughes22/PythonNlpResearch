db.getCollection('SC_TAGGING_VD_CRF_LBL_POWERSET_HYPERPARAM_OPT')
.aggregate(
[{
    $project: { 
            weighted_f1_score: "$WEIGHTED_MEAN_CONCEPT_CODES.f1_score",
            //macro_f1: "$MACRO_F1",
            micro_f1_score: "$MICRO_F1.f1_score",
            window_size: "$parameters.window_size",
            feats: "$parameters.extractors",
            count: { $size:"$parameters.extractors" },
            feature_possible_states      : "$parameters.feature_possible_states",
            feature_possible_transitions : "$parameters.feature_possible_transitions",
            c2 : "$parameters.c2",
            asof: 1,
            //parameters: "$parameters",
            "_id":0
    }
},
{
    $match:{
        //count: {  $eq:2 },
        //window_size: { $eq:9 }
        //micro_f1_score: { $exists : true }
    }
},
{    $sort:{
        //count: 1,
        //"macro_f1":-1,
        //"weighted_f1_score":-1,
        "micro_f1_score":-1,
        //asof: -1
        
    }
},
])