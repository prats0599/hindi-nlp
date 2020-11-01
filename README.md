This repo includes building a state of the art language modelling and
text classification architectures. We use ULMFiT which utilizes transfer learning on
AWD-LSTMS to build great classifiers in any language needed. The idea is to first train a language model
that predicts the next word given a set of words in that language(hindi here). We then extract the Wikipedia articles
in that language and train the model; Finetune the language model on the classification dataset, save the encoder
and use that as our classifier(after adding the fully connected layers) for sentiment analysis.  
To improve results, we build another language model that predicts the previous word given a set of words, i.e. we feed
the data in the reverse order and ask the model to predict word 1, given words n to 2, where n>2. We follow the same steps as above
and after implementing the classification model, we simply ensemble to the two models and voila!

## Datasets  
1. [BBC News Articles](https://github.com/AI4Bharat/indicnlp_corpus#publicly-available-classification-datasets) : Sentiment analysis corpus for Hindi documents extracted from BBC news website.  

2. [IITP Product Reviews](https://github.com/AI4Bharat/indicnlp_corpus#publicly-available-classification-datasets) : Sentiment analysis corpus for product reviews posted in Hindi.  

3. [IITP Movie Reviews](https://github.com/AI4Bharat/indicnlp_corpus#publicly-available-classification-datasets) : Sentiment analysis corpus for movie reviews posted in Hindi.  


## Notebooks

[nn-hindi](nn-hindi.ipynb): Contains code for Model 1 and ensembling.  
[nn-hindi-bwd](nn-hindi-bwd.ipynb): Code to train the models that predicts text backwards(Model 2).  
[bbc-hindi](bbc-hindi.ipynb): Same code as nn-hindi.ipynb, but just for the bbc-articles dataset.   


## Results  

  ### Language Model Perplexity(on validation dataset which is randomly split)
  | Architecture | Dataset                 |  Accuracy |
  | -------------|-------------------------|-----------|
  | ULMFiT       | Wikipedia-hi            |    30.17  |
  | ULMFiT       | Wikipedia-hi(backwards) |    29.25  |

  ### Classification Metrics(on test set)
  |        Dataset           | Accuracy(Model 1) |   MCC(Model 1)    | Accuracy(Model 2) |   MCC(Model 2)   | Accuracy(ensemble) |   MCC(ensemble)    |
  |--------------------------|-------------------|-------------------|-------------------|------------------|--------------------|--------------------|
  | BBC Articles(14 classes) |      79.79        |       72.58       |       78.75       |      71.15       |       84.39        |       79.13        |
  |  IITP movie Reviews      |      58.39        |       38.34       |       61.94       |      43.68       |                    |                    |        
  |  IITP Product Reviews    |      72.08        |       54.19       |       75.90       |      59.83       |                    |                    |

Just by ensembling, we have outperformed classification benchmarks mentioned in this [repository](https://github.com/goru001/nlp-for-hindi).  

NOTE: MCC metric mentioned in the table refers to [matthews correlation coefficient](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6413-7#:~:text=The%20Matthews%20correlation%20coefficient%20(MCC)%2C%20instead%2C%20is%20a,both%20to%20the%20size%20of).  

## Download Pretrained models
The pretrained language models(both forward and backward) are available to download [here](https://drive.google.com/drive/folders/1k0cZ3e8_MPUhn3WWhd3klg-lfEEzNAZ5?usp=sharing).


## Future Work  
- [x] Train second model on IITP movie reviews and product reviews datasets.  
- [ ] Ensemble the other two models
- [ ] Make a separate notebook for each dataset.
- [ ] Experiment using transformers instead of LSTMs and compare results.  



### The full article on how to create your own SOTA model for language modelling & sentiment analysis is available [here](https://prats0599.medium.com/building-a-state-of-the-art-text-classifier-for-any-language-you-want-fe3ebbdab5c9).  
