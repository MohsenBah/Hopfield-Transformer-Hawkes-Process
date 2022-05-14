# New-Hawkes

Modeling Multivariate Hopfield-Transformer Hawkes Process: Application to Sovereign Credit Default Swaps

Hawkes process was evolved so that the past events contribute to the occurrence time of
future events by self-exciting or mutually exciting. However, many real-world data do
not follow the Hawkes process’s assumptions (i.e., positivity, additivity, and exponential
decay) and become more complex to be modeled by the traditional Hawkes processes, so the
neural Hawkes process was developed to tackle the challenges. However, Recurrent Neural
Networks (RNN) fail to capture long-term dependencies among multiple point processes, and
Transformer Hawkes processes only address temporal characteristics of Hawkes processes. In
this thesis, we proposed a combination of neural networks and Hawkes processes to tackle the
aforementioned challenges and to capture contagious effects among different points processes.
First, we made substantial modifications to the Transformer Hawkes process by utilizing two
encoders, which include two Multi-Head attention modules: 1) event significance attention and
2) temporal attention. Then, to improve this model, the Modern Hopfield Neural Network was
incorporated to better assign the attention to the test set by appending the decoder layer to the
previous modified encoder layers. Credit Default Swap data for ten European countries were
tested, and the results revealed that modeling the contagious effect ameliorates the prediction
performance.

Mohsen Bahremani - Master thesis - Wilfrid Laurier Univeristy, Waterloo, ON.
https://scholars.wlu.ca/etd/2417/
