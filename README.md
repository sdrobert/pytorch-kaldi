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
  ``` sh
    steps/align_fmllr.sh --nj 4 data/dev data/lang exp/tri3 exp/tri3_ali_dev
    # test ali won't actually be used, except to verify the length of output
    # features and as a placeholder in run_nn.py
    steps/align_fmllr.sh --nj 4 data/test data/lang exp/tri3 exp/tri3_ali_test
  ```

2. Construct symbolic links to the WSJ recipe's *steps* and *utils*
   directories. If you've properly positioned this folder in TIMIT's recipe
   directory, then the following commands should suffice from this folder:
   ``` sh
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
    ``` shell
    ./run_exp.sh exp/fbank_liGRU liGRU fbank
    ```
    Which builds a configuration based on the partial configurations of
    *conf/partial/fbank.cfg* and *conf/partial/liGRU.cfg* and will store
    results in *exp/fbank_liGRU*. More configuration names can be added as
    arguments to *run_exp.sh*, resulting in those configurations being appended
    into one large config
  - Alternatively, a whole bunch of train/decode cycles can be performed
    using the script *run_exp_series.sh*. Arguments are comma-delimited lists
    of the names of partial configurations. Each comma-delimited argument
    represents a factor; the product of all factors will be run. e.g.
    ``` shell
    ./run_exp_series --num_trials 2 a,b c,d e
    ```
    is identical to running
    ``` shell
    ./run_exp.sh --seed 1 exp/a_c_e_1 a c e
    ./run_exp.sh --seed 2 exp/a_c_e_2 a c e
    ./run_exp.sh --seed 1 exp/a_d_e_1 a d e
    ./run_exp.sh --seed 2 exp/a_d_e_2 a d e
    ./run_exp.sh --seed 1 exp/b_c_e_1 b c e
    ./run_exp.sh --seed 2 exp/b_c_e_2 b c e
    ./run_exp.sh --seed 1 exp/b_d_e_1 b d e
    ./run_exp.sh --seed 2 exp/b_d_e_2 b d e
    ```
    Convenient!

5. Get phone error rates for your experiments. Kaldi's regular strategy is
   to decode utterances using a variety of language model weights, then use
   the best ones for the development and test sets independently. For this
   behaviour, run the *RESULTS* script. Alternatively,
   *RESULTS_tunedevonly* finds the best development set language model
   weight and reports the error rate of the test set using *that* weight.
   Additionally, you can pipe the output of the results script to *stats.py*
   to perform a MANOVA analysis.


## References

[1] M. Ravanelli, P. Brakel, M. Omologo, Y. Bengio, "Improving speech recognition by revising gated recurrent units", in Proceedings of Interspeech 2017. [ArXiv](https://arxiv.org/abs/1710.00641)

[2] M. Ravanelli, P. Brakel, M. Omologo, Y. Bengio, "Light Gated Recurrent Units for Speech Recognition", in IEEE Transactions on
Emerging Topics in Computational Intelligence. [ArXiv](https://arxiv.org/abs/1803.10225)

[3] M. Ravanelli, "Deep Learning for Distant Speech Recognition", PhD Thesis, Unitn 2017. [ArXiv](https://arxiv.org/abs/1712.06086)

[4] D. Serdyuk, R. Nan Ke, A. Sordoni, A. Trischler, C. Pal, Y. Bengio, "Twin Networks: Matching the Future for Sequence Generation", ICLR 2018 [ArXiv](https://arxiv.org/pdf/1708.06742.pdf)
