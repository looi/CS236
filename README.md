# CS 236 Project

* Birds dataset provided by MirrorGAN zip file
* Some data loading code from MirrorGAN used, heavily modified
* Merged with CS 236 starter code
* PixelCNN++ trained to generate birds, both unconditional and conditional
* Facebook InferSent for embedding sentence to word
  * We could also compare with BERT provided in the CS 236 starter code. But it takes the average of word embeddings, so it doesn't seem like the best choice for sentences
  * Another option: https://github.com/UKPLab/sentence-transformers
