This repo includes building a state of the art language modelling and
text classification architectures. We use ULMFiT which utilizes transfer learning on
AWD-LSTMS to build great classifiers in any language needed. The idea is to first train a language model
that predicts the next word given a set of words in that language(hindi here). We then extract the wikipedia articles
in that language and train the model; Finetune the language model on the classification dataset, save the encoder
and use that as our classfier(after adding the fully connected layers) for sentiment analysis.  


    ## Future Work
    Experiment using transformers instead of LSTMs and compare results.  
    

### The full article is available at [medium]().  
