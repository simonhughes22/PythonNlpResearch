// NOTE: if we set tag frequency to 0, this still mildly out-performs the 
// logistic regression window-based mode. However need to compare performance of this model over different
// nuk_iteration settings, as that may be just becaause we have over-tuned Perceptron relative to the window based LR model
// Once we add in tag history, we see a v SMALL increase in performance. So on this tasks, this does not seem to add
// much predictive power, but it works better using an iterative (online) appraoch to tagging that a batch process (c.f. CRF model)
// as predicted
db.getCollection('CB_TAGGING_VD_AVG_PERCEPTRON_MULTICLASS')
.aggregate(
[{
    $project: { 
            //weighted_f1_score: "$WEIGHTED_MEAN_CONCEPT_CODES.f1_score",
            //macro_f1: "$MACRO_F1",
            micro_f1_score: "$MICRO_F1.f1_score",
            window_size: "$parameters.window_size",
            tag_history: "$parameters.tag_history",
            num_iterations: "$parameters.num_iterations",
            //feats: "$parameters.extractors",
            count: { $size:"$parameters.extractors" },
            "_id":0,
            //params: "$parameters",
            asof: 1,            
    }
},
{
    $match:{
        //count: {  $eq:2 },
        //window_size: { $eq:9 }
        //tag_history: {$eq: 0},
        micro_f1_score: { $exists : true }
    }
},
{    $sort:{
        "micro_f1_score":-1,
        //asof: -1
        
    }
},
])