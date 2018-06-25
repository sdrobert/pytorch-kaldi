#! /usr/bin/env bash
# Run a series of experiments

. path.sh
. cmd.sh

exp_dir=exp
log_dir=exp/log
nnet_archs=liGRU
feats=mfcc,kaldi,fbank,gbank,sifbank,sigbank
num_trials=5

. utils/parse_options.sh

set -e

IFS=, read -a nnet_arch_list <<< ${nnet_archs}
IFS=, read -a feats_list <<< ${feats}

for arch in ${nnet_arch_list[@]}; do
  for feat in ${feats_list[@]}; do
    for trial in $(seq 1 ${num_trials}); do
      echo "Beginning trial ${trial} of nnet ${arch} using ${feat} feats"
      ./run_one_exp.sh \
        --seed "${trial}" \
        --log-dir "${log_dir}/${arch}_${feat}_${trial}" \
        "${arch}" "${feat}" "exp/${arch}_${feat}_${trial}"
      echo "Done trial ${trial} of nnet ${arch} using ${feat} feats"
    done
  done
done
