# CS 236 Project - Caption-to-Image Conditional Generative Modeling

* Birds dataset provided by [MirrorGAN](https://github.com/qiaott/MirrorGAN) zip file
* Some data loading code from MirrorGAN used, heavily modified
* Merged with CS 236 starter code
* PixelCNN++ trained to generate birds, both unconditional and conditional
  * Unused in final paper, because results are not good even after training for days.
* ACGAN (Auxiliary Classifier GAN) with several different conditioning:
  * Unconditional
  * One-hot vector (bird species)
  * Facebook InferSent for embedding sentence to word
  * BERT average of word embedding vectors (provided in the CS 236 starter code).
    * Doesn't seem like the best choice for sentences
  * [Sentence-BERT](https://github.com/UKPLab/sentence-transformers)
* MirrorGAN
  * Baseline STEM model (bidirectional GRU)
  * With BERT as alternative conditionong
