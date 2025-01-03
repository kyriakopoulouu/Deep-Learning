# Deep-Learning
This project was implemented as part of the postgraduate studies in Computational Intelligence and Digital Media. 
The purpose of this project was to familiarize ourselves with the basic types of deep neural network architectures and their training to solve problems of our choice. The architectures and training methods that were required to be explored included the following:

1) Deep Convolutional Neural Networks (CNNs)
2) Deep Recurrent Neural Networks (RNNs) or Transformers
3) Deep Reinforcement Learning (DRL)

1. For Deep Convolution Neural Networks (CNNs), a dataset for Facial Emotion Recognition was selected through  Kaggle. This dataset consists of 35,887 images, each 48x48 pixels in grayscale. Each image represents a different
emtotion. The emotions in the images are : angry, disgust, fear, happy, neutral, sad, and surprise. We dealt with a multi-class image classification problem involving seven classes.

2. For Deep Recurrent Neural Networks (RNNs), a dataset for Fake News Detection Analysis was  was selected through Kaggle. This dataset contains a collection of 20800 English articles. Its structure is (20800.5),
i.e. 20800 rows and 5 column-attributes. Specifically, the characteristics of the columns include:

    id: unique identifier for a news article
 
   title: title of news article
 
   author: author of the news article
 
   text: the text of the article. may be incomplete
 
   tag: a tag that marks the article as potentially untrustworthy
 
   1: unreliable
 
   0: reliable
3. For Deep Reinforcement Learning (DRL), the Atari Mario Bros game was selected from the Gymnasium library. Using the Stable-Baselines3 library and the Proximal Policy Optimization (PPO) algorithm, the agent was trained, specifically employing the CNNPolicy. Initially, the model was trained with its default parameters. Subsequently, a custom convolutional network was added, followed by experimentation with modifications and enhancements to this network to improve its performance.
