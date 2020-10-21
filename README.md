This repo includes building a state of the art language modelling and
text classification architectures. We use ULMFiT which utilizes transfer learning on
AWD-LSTMS to build great classifiers in any language needed. The idea is to first train a language model
that predicts the next word given a set of words in that language(hindi here). We then extract the Wikipedia articles
in that language and train the model; Finetune the language model on the classification dataset, save the encoder
and use that as our classifier(after adding the fully connected layers) for sentiment analysis.  
To improve results, we build another language model that predicts the previous word given a set of words, i.e. we feed
the data in the reverse order and ask the model to predict word 1, given words n to 2, where n>2. We follow the same steps as above
and after implementing the classification model, we simply ensemble to the two models and voila!

## Results  

  ### Language Model Perplexity(on validation dataset which is randomly split)
  | Architecture | Dataset                 |  Accuracy |
  | -------------|-------------------------|-----------|
  | ULMFiT       | Wikipedia-hi            |    30.17  |
  | ULMFiT       | Wikipedia-hi(backwards) |    29.25  |

  ### Classification Metrics(on test set)
  | Dataset                  |       Model 1       |  Model 2(backwards)  |        Ensemble      |
  |--------------------------|----------|----------|----------|-----------|----------|-----------|
  |                          | Accuracy |   MCC    | Accuracy |   MCC     | Accuracy |   MCC     |
  |--------------------------|----------|----------|----------|-----------|----------|-----------|
  | BBC Articles(14 classes) |   79.79  |  72.58   |          |           |  84.39   |  79.13    |
  |  IITP movie Reviews      |   58.39  |  38.34   |          |           |          |           |          
  |  IITP Product Reviews    |   72.08  |  54.19   |          |           |          |           |





## Future Work  
Train second model on IITP movie reviews and product reviews datasets.  
Experiment using transformers instead of LSTMs and compare results.  



### The full article on how to create your own SOTA model for language modelling & sentiment analysis is available [here]().  
