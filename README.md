# pytorch-kaldi
pytorch-kaldi is a public repository for developing state-of-the-art DNN/RNN hybrid speech recognition systems. The DNN part is managed by pytorch, while feature extraction, label computation, and decoding are performed with the kaldi toolkit.


## Introduction:
This project releases a collection of codes and utilities to develop state-of-the-art DNN/RNN hybrid speech recognition systems. The DNN/RNN part is implemented in pytorch, while feature extraction, alignments, and decoding are performed with the Kaldi toolkit.  The current version of the provided system has the following features:
- Supports different types of NNs (e.g., *MLP*, *RNN*, *LSTM*, *GRU*, *Minimal GRU*, *Light GRU*) [1,2,3]
- Supports  recurrent dropout
- Supports  batch and layer normalization
- Supports unidirectional/bidirectional RNNs
- Supports  residual/skip connections
- Supports  twin regularization [4]
- python2/python3 compatibility
- multi-gpu training
- recovery/saving checkpoints
- easy interface with kaldi.

The provided solution is designed for large-scale speech recognition experiments on both standard machines and HPC clusters. 

## Prerequisites:
- Linux is required (we tested our release on Ubuntu 17.04 and various versions of Debian).

- We recommend to run the codes on a GPU machine. Make sure that the cuda libraries (https://developer.nvidia.com/cuda-downloads) are installed and correctly working. We tested our system on cuda 9.0, 9.1 and 8.0.
Make sure that python is installed (the code is tested with python 2.7 and python 3.6). Even though not mandatory, we suggest to use Anaconda (https://anaconda.org/anaconda/python).

- If not already done, install pytorch (http://pytorch.org/). We tested our codes on pytorch 0.3.0 and pytorch 0.3.1. Older version of pytorch are likely to rise errors. To check your installation, type “python” and, once entered into the console, type “import torch”. Make sure everything is fine.

- If not already done, install Kaldi (http://kaldi-asr.org/). As suggested during the installation, do not forget to add the path of the Kaldi binaries into $HOME/.bashrc. As a first test to check the installation, open a bash shell, type “copy-feats” and make sure no errors appear.

- Clone this repository into the TIMIT experiment directory *${KALDI_ROOT}/egs/timit*

- The implementation of the RNN models sorts the training sentences according to their length. This allows the system to minimize the need of zero padding when forming minibatches. The duration of each sentence is extracted using *sox*. Please, make sure it is installed (it is only used when generating the feature lists in *create_chunk.sh*)

- Install pydrobert-kaldi and pydrobert-speech from pip or anaconda

## How to run a TIMIT experiment:
Even though the code can be easily adapted to any speech dataset, in the
following part of the documentation we provide an example based on the popular
TIMIT dataset.

1. Run the Kaldi s5 baseline of TIMIT. This step is necessary to compute
   features and labels later used to train the pytorch networks. In particular:
  - go to *$KALDI_ROOT/egs/timit/s5*
  - add the line `--snip-edges=false` to *conf/mfcc.conf* and *conf/fbank.conf*
  - run the script *run.sh* sourcing `path.sh` before
  - from the *s5* dir, call
  ``` 
  steps/align_fmllr.sh --nj 4 data/dev data/lang exp/tri3 exp/tri3_ali_dev

  steps/align_fmllr.sh --nj 4 data/test data/lang exp/tri3 exp/tri3_ali_test
  ``` 

2. Construct symbolic links to the WSJ recipe's *steps* and *utils*
   directories. If you've properly positioned this folder in TIMIT's recipe
   directory, then the following commands should suffice from this folder:
   ```
    ln -s ../../wsj/s5/utils utils
    ln -s ../../wsj/s5/steps steps
   ```

3. Generate features, split into chunks, and partially construct config files
  - Go back to the directory of this readme
  - Run the script *build_feats.sh*. Have a look at the source for the various
    ways the features can be configured


4. Run the experiment(s)
  - A single train/decode cycle can be executed using the *run_one_exp.sh*
    command, for example:
    ``` 
    ./run_exp.sh fbank liGRU exp/fbank_liGRU
    ```
    Which builds a configuration based on the partial configurations of
    *conf/partial/fbank.cfg* and *conf/partial/liGRU.cfg* and will store
    results in *exp/fbank_liGRU*.
  - Alternatively, a whole bunch of train/decode cycles using the script
    *run_exp_series.sh*

5. Get phone error rates for your experiments. Kaldi's regular strategy is
   to decode utterances using a variety of language model weights, then use
   the best ones for the development and test sets independently. For this
   behaviour, run the *RESULTS.sh* script. Alternatively,
   *RESULTS_tunedevonly.sh* finds the best development set language model
   weight and reports the error rate of the test set using *that* weight.


## TIMIT Results:

The results reported in each cell of the  table are the average *PER%* performance obtained  on the test set  after running five ASR experiments with different initialization seeds. We believe that averaging the performance obtained with different initialization seeds is crucial  for TIMIT, since the natural performance variability might completely hide the experimental evidence.  

The main hyperparameters of the models (i.e., learning rate, number of layers, number of hidden neurons, dropout factor) have been optimized through a grid search performed on the development set (see the config files in *conf/baselines* for an overview on the hyperparameters adopted for each NN). 

The RNN models are bidirectional, use recurrent dropout, and batch normalization is applied to feedforward connections. 

| Model  | mfcc | fbank | fMLLR | 
| ------ | -----| ------| ------| 
|  Kaldi DNN Baseline | -----| ------| 18.5 |
|  MLP  | 18.2 ± 0.19| 18.6 ± 0.24| 16.9 ± 0.19| 
|LSTM| 15.7 ± 0.27 | 15.1 ± 0.26 |14.7 ± 0.16 | 
|GRU| 16.0 ± 0.13| 15.3 ± 0.27 |  15.3 ± 0.32| 
|M-GRU| 16.1  ± 0.28| 15.4 ± 0.11|  15.2 ± 0.23| 
|li-GRU| **15.5**  ± 0.33| **14.6** ± 0.10|  **14.6** ± 0.32| 


The RNN architectures are significantly better than the MLP one. In particular, the Li-GRU model (see [1,2] for more details) performs slightly better that the other models. As expected fMLLR features lead to the best performance. The performance of  *14.6%* obtained with our best fMLLR system is, to the best of our knowledge, one of the best results so far achieved with the TIMIT dataset.

For comparison and reference purposes,  you can find the output results obtained by us in the folders  *exp/our_results/TIMIT_{MLP,RNN,LSTM,GRU,M_GRU,liGRU}*. 


## Brief Overview of the Architecture

The main script to run experiments is *run_exp.sh*.  The only parameter that it takes in input is the configuration file, which contains a full description of the data, architecture, optimization and decoding step. The user can use the variable *$cmd* for submitting jobs on HPC clusters.

Each training epoch is divided into many chunks.  The pytorch code *run_nn_single_ep.py* performs training over a single chunk and provides in output a model file in *.pkl* format and a *.info* file (that contains various information such as the current training loss and error). 

After each training epoch, the performance on the dev-set is monitored. If the relative performance improvement  is below a given threshold, the learning rate is decreased by an halving factor. The training loop is iterated for the specified number of training epochs. When training is finished, a forward step is carried on for generating the set of posterior probabilities that will be processed by the kaldi decoder. 

After decoding, the final transcriptions and scores are available in the output folder. If, for some reason, the training procedure is interrupted the process can be resumed starting from the last processed chunk.



## Adding customized DNN models
One can easily write its own customized DNN model and plugs it into neural_nets.py. Similarly to the models already implemented, the user has to write a *init* method for initializing the DNN parameters and a forward method. The forward method should take in input the current features *x* and the corresponding labels *lab*. It has to provide at the output the loss, the error and the posterior probabilities of the processed minibatch.  Once the customized DNN has been created, the new model should be imported into the *run_nn_single_ep.py* file in this way:

``` 
from neural_nets import mydnn as ann
``` 

It is also important to properly set the label *rnn=1* if the model is a RNN model and *rnn=0* if it is a feedforward DNNs. Note that RNN and feed-forward models are based on different feature processing (for RNN models  the features are ordered according to their length, for feed-forward DNNs the features are shuffled.)


## References

[1] M. Ravanelli, P. Brakel, M. Omologo, Y. Bengio, "Improving speech recognition by revising gated recurrent units", in Proceedings of Interspeech 2017. [ArXiv](https://arxiv.org/abs/1710.00641)

[2] M. Ravanelli, P. Brakel, M. Omologo, Y. Bengio, "Light Gated Recurrent Units for Speech Recognition", in IEEE Transactions on
Emerging Topics in Computational Intelligence. [ArXiv](https://arxiv.org/abs/1803.10225)

[3] M. Ravanelli, "Deep Learning for Distant Speech Recognition", PhD Thesis, Unitn 2017. [ArXiv](https://arxiv.org/abs/1712.06086)

[4] D. Serdyuk, R. Nan Ke, A. Sordoni, A. Trischler, C. Pal, Y. Bengio, "Twin Networks: Matching the Future for Sequence Generation", ICLR 2018 [ArXiv](https://arxiv.org/pdf/1708.06742.pdf)
  



