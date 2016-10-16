db.getCollection('CB_TAGGING_VD_FEAT_SELECTION').aggregate(
[ 
    {
        $project: { 
            //weighted_f1_score: "$WEIGHTED_MEAN_CONCEPT_CODES.f1_score",
            micro_f1_score: "$MICRO_F1.f1_score",
            window_size: "$parameters.window_size",
            //params: "$parameters",
            //feats: "$parameters.extractors",
            //minscore: "$parameters.MIN_SCORE",
            //maxrules: "$parameters.MAX_RULES",
            //base_tagger: "$parameters.BASE_TAGGER",
            count: { $size:"$parameters.extractors" },
            //asof: 1,
            "_id":1
        }
    },
    
    {
        $match:{ 'micro_f1_score': { $exists : true } }
    },
    
    {     $group: {
                _id: {
                    window_size  : "$window_size",
                    //feat_count: "$count",
                },
                max_feats: { $max : "$count" },
                min_feats: { $min : "$count" },
                //count: { $size:"$parameters.extractors" }
                num_rows: { $sum: 1 }
        }            
    },
    {   $project: {
            window_size : "$_id.window_size",
            max_feats  :  1,
            min_feats  :  1,
            num_rows   :  1,
            _id:0
        }
    },
    {   $sort: {
            "window_size" : -1
        }               
    }
])