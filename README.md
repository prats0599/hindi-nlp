This repo includes building a state of the art language modelling and
text classification architectures. We use ULMFiT which utilizes transfer learning on
AWD-LSTMS to build great classifiers in any language needed. The idea is to first train a language model
that predicts the next word given a set of words in that language(hindi here). We then extract the wikipedia articles
in that language and train the model; Finetune the language model on the classification dataset, save the encoder
and use that as our classfier(after adding the fully connected layers) for sentiment analysis.  

## Results  
  
  ### Language Model Perplexity(on validation dataset which is randomly split) 
  | Architecture | Dataset             |  Accuracy |
  | -------------|---------------------|-----------|
  | ULMFiT       | Wikipedia-hi        |    30.    |   
     
  ### CLassfication Metrics  
   | Dataset | Accuracy             |  MCC         |
  | -------------|---------------------|-----------|
  | ULMFiT       | Wikipedia-hi        |    30.    |   
  
  
  
  
  
## Future Work  
Experiment using transformers instead of LSTMs and compare results.  
    

### The full article on how to create your own sota model for language modelling & sentiment analysisis is available [here]().  
