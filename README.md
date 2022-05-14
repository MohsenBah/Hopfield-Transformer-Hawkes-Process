# Hopfield-Transformer Hawkes Process

Modeling Multivariate Hopfield-Transformer Hawkes Process: Application to Sovereign Credit Default Swaps

# Code and Data

Data: We collected daily CDS rates for contracts with 1-year to 10-year durations from 3 November 2008 to 28 February 2012 from the platform, **Thomson Reuters Datastream** (Datastream International,(May 6, 2021), Credit Default SWAP[online], Available: Refinitiv/CDS)

Code: **Hawkes_process.ipynb** initially deals with data preprocessing and visualization. Then the model runs by **bash run.sh**, which includes neural network parametes definition. The model is written by PyTorch.


# Abstract

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


# Recommended Citation

Bahremani, Mohsen, "Modeling Multivariate Hopfield-Transformer Hawkes Process: Application to Sovereign Credit Default Swaps" (2021). Theses and Dissertations (Comprehensive). 2417.
https://scholars.wlu.ca/etd/2417 

```
@article{bahremani2021modeling,
  title={Modeling Multivariate Hopfield-Transformer Hawkes Process: Application to Sovereign Credit Default Swaps},
  author={Bahremani, Mohsen},
  year={2021},
  publisher={Wilfrid Laurier University}
}
```
