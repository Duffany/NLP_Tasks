# NLP_Tasks
NLP notebooks featuring : Sentiment Analysis, Language models, Attention concept

## Exercise 1: Improving the performances of a sentiment analysis classifier
From the notebook of day 2, try to improve the performances of the sentiment analysis classifier by trying the following changes in the model/algorithm/preprocessing procedure.
To evaluate the performances, look at the validation accuracy.

1. Preprocessing: in the original dataset, remove stop_words and do some stemming (see https://www.nltk.org/howto/stem.html documentation)

- Does it improve performance ?

2. Model architecture changes:

- Replace the trainable embedding layer by an embedding layer which loads GloVe embeddings (see TP of day #1).
- Use several layers of LSTM
- Add a Dense Layer after the LSTM and before the final Dense Projection Layer
- Use a bi-directionnal LSTM using https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional
- Take the average of the sequence of hidden states as the sentence encoding (instead of the final hidden state)
- Do you find an architecture that improves the performance of the classifier ?

3. Training algorithm / hyper-parameters search

- Add gradient clipping to limit exploding gradients: https://www.tensorflow.org/api_docs/python/tf/clip_by_norm
- Do hyperparameter-search on model dimensions, dropout rate, training hyper-parameters (learning rate, batch size) to reach a better val accuracy.
- Which set of hyper-parameters work the best ?


## Exercise 2: Learning a Language Model from scratch
Learn a Language Model from the text "metamorphosis.txt" (dataset of notebook of day 1) and generate text with it.

- Preprocess your dataset for your needs (split into sentences, tokenize, clean text, build vocabulary, encode sentences)
- Create tensorflow datasets and dataloader from the processed datasets
- Build a RNN Language Model (see notebook of day #3) and train it
- Compute its perplexity over the test set: can we obtain a perplexity around or inferior to 3 in the validation set ?
- Generate text with it using nucleus sampling (or top-k nucleus sampling): See this chunk of code as an help to implement the method: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317 and print the generated text.
- Bonus Question: Take a random sentence of the test dataset, generate text with the trained language model, and compare it with text generated with GPT-2. See hugging face tutorial: https://huggingface.co/blog/how-to-generate

## Exercise 3: Train a Sequence to Sequence Model with Attention on the ROC Story Dataset
Using the ROC Story Dataset of day #3 (lien ici), build a sequence to sequence model with attention that takes as input sentence the sentence #1 (input of the encoder), and takes as target sentence (input of the decoder), the sentence #2.
This creates an encoder-decoder model for story continuation. For the model, you can use either:

- The Seq2Seq RNN Model with attention of notebook of day #4. You can tweak the model, for example using a Multiplicative Attention instead of an Additive Attention

- A Transformer model. There is a great tutorial and transformer implementation in tensorflow here: https://www.tensorflow.org/text/tutorials/transformer
