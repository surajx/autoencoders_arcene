## Modeling High-Dimensional Classification Problems Using Deep Learning

Deep learning methods have been successfully applied to convert high-dimensional
to low-dimensional codes . These low-dimensional codes are essentially higher-
level features that can provide good discriminability for classification tasks. 
In this study we train a deep network for feature extraction and classification
on the 10,000 dimensional ARCENE dataset. The unsupervised pre-training for
feature extraction is done by a stacked autoencoder (SAE), and the subsequent
supervised logistic regression by a softmax classifier. We then compare the
balanced error rate (BER) performance measure of the deep network with that of
a feedforward neural network. The report also highlights why a Bayesian neural
network (BNN) was considered but not selected for the study.

Full report: [paper.pdf](https://github.com/surajx/autoencoders_arcene/raw/master/autoencoders_arcene/paper.pdf)
