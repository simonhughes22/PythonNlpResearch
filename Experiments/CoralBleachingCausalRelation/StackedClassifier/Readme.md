Please see /notebooks/Keras

Note that for the other Crel classifiers, I changed the code to map the Causer:50 and Result:50 type tags
to their underlying codes, as parts of the data were missing the underlying labels. You will find code for
this in the Stacked Model training and in the Shift Reducer parser training, as both of those methods rely
on the underlying concept code predictions. This was NOT done for the RNN though, because  that attempts to
predict the causal relations directly, and so would not be impacted by this change.

For the concept code work, this was not done as it was not discovered until the work was commenced for RQ2. 
It likely does not impact or invalidate the research, as all concept code tagging methods were compared on 
the same data (albeit lacking some concept code examples) and so it is still a valid comparison.