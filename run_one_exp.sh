#!/bin/bash

##########################################################
# pytorch_speechMLP
# Mirco Ravanelli
# Montreal Institute for Learning Algoritms (MILA)
# University of Montreal
# Feb 2018
##########################################################
. env.sh
. path.sh
. cmd.sh

cmd="${train_cmd}"
N_ep=24
lr=0.0008
optimizer=rmsprop
halving_factor=0.5
improvement_threshold=0.001
batch_size=8
save_gpumem=0
partial_cfg_folder=conf/partial
log_dir=exp/log
s5_dir="${KALDI_ROOT}/egs/timit/s5"
graph_dir="${s5_dir}/exp/tri3/graph"
data_dir=data
seed=0
ctype=ord

. utils/parse_options.sh

if [ $# -lt 1 ]; then
  echo -e "Usage: ./run_one_exp.sh [options] <out-folder> [<cfg1> [<cfg2> ...]]"
  echo -e "e.g. ./run_one_exp.sh exp/liGRU_fbank liGRU fbank"
  exit 1
fi

out_folder=$1
exp_name=$(basename "${out_folder}")

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

cfg_file="${tmpdir}/init.cfg"

> "${cfg_file}"
for cfg_name in "${@:2}"; do
  cat "${partial_cfg_folder}/${cfg_name}.cfg" >> "${cfg_file}"
done

echo "
[DEFAULT]
ctype=${ctype}

[optimization]
N_ep=${N_ep}
lr=${lr}
optimizer=${optimizer}
halving_factor=${halving_factor}
improvement_threshold=${improvement_threshold}
batch_size=${batch_size}
save_gpumem=${save_gpumem}

[architecture]
seed=${seed}

[data]
out_folder=\"${out_folder}\"
" >> "${cfg_file}"

function write_conf {
    # Writing the single-iteration config file

    echo "[DEFAULT]" > $conf_file
    echo "ctype = ${ctype}" >> $conf_file
    echo " " >> $conf_file

    echo "[todo]" >> $conf_file
    echo "do_training=$do_training" >> $conf_file
    echo "do_eval=$do_eval" >> $conf_file
    echo "do_forward=$do_forward" >> $conf_file
    echo " " >> $conf_file

    echo "[data]" >> $conf_file
    echo "fea_scp=$fea_chunk" >> $conf_file
    echo "fea_opts=$fea_opts" >> $conf_file
    echo "lab_folder=$lab_folder" >> $conf_file
    echo "lab_opts=$lab_opts" >> $conf_file
    echo "pt_file=$pt_file" >> $conf_file
    echo "count_file=$count_file" >> $conf_file
    echo "out_file=$out_file" >> $conf_file
    echo " " >> $conf_file

    echo "[architecture]" >> $conf_file
    echo "NN_type=$NN_type" >> $conf_file
    echo "cnn_pre=$cnn_pre" >> $conf_file
    echo "hidden_dim=$hidden_dim" >> $conf_file
    echo "N_hid=$N_hid" >> $conf_file
    echo "drop_rate=$drop_rate" >> $conf_file
    echo "use_batchnorm=$use_batchnorm" >> $conf_file
    echo "use_laynorm=$use_laynorm" >> $conf_file
    echo "cw_left=$cw_left" >> $conf_file
    echo "cw_right=$cw_right" >> $conf_file
    echo "seed=$seed" >> $conf_file
    echo "use_cuda=$use_cuda" >> $conf_file
    echo "bidir=$bidir" >> $conf_file
    echo "resnet=$resnet" >> $conf_file
    echo "skip_conn=$skip_conn" >> $conf_file
    echo "act=$act" >> $conf_file
    echo "act_gate=$act_gate" >> $conf_file
    echo "resgate=$resgate" >> $conf_file
    echo "minimal_gru=$minimal_gru" >> $conf_file
    echo "cost=$cost" >> $conf_file
    echo "twin_reg=$twin_reg" >> $conf_file
    echo "twin_w=$twin_w" >> $conf_file
    echo "multi_gpu=$multi_gpu" >> $conf_file
    echo "block_type=$block_type" >> $conf_file
    echo "channel_factor=$channel_factor" >> $conf_file
    echo "ds_factor=$ds_factor" >> $conf_file
    echo "group_counts=$group_counts" >> $conf_file
    echo "init_channels=$init_channels" >> $conf_file
    echo "conv_type=$conv_type" >> $conf_file
    echo "kernel_size=$kernel_size" >> $conf_file
    echo "conv_channel_sizes=$conv_channel_sizes" >> $conf_file
    echo " " >> $conf_file

    echo "[optimization]" >> $conf_file
    echo "lr=$lr" >> $conf_file
    echo "optimizer=$optimizer" >> $conf_file
    echo "batch_size=$batch_size" >> $conf_file
    echo "save_gpumem=$save_gpumem" >> $conf_file
}

# Parsing cfg file
source <(grep = $cfg_file)
if [ -z "${te_data_folder}" ]; then
  echo -e "None of your config files set te_data_folder. This likely means
you forgot to pass a config file for features"
  exit 1
fi
IFS=, read -a tr_fea_scp_list <<< $tr_fea_scp
IFS=, read -a dev_fea_scp_list <<< $dev_fea_scp
IFS=, read -a te_fea_scp_list <<< $te_fea_scp

# creating output folder
mkdir -p "${out_folder}"

# Initialization
pt_file='none'

# Number of training chunks
N_ck=${#tr_fea_scp_list[@]}

echo 'Training...'
mkdir -p "${log_dir}"
# Main Training Loop (Training+Eval)
for epoch in $(seq -w 1 $N_ep); do
  if [ "$epoch" -gt "1" ]; then
    err_dev_prev=$err_dev
  fi

  for chunk in $(seq -w 0 "$(($N_ck-1))"); do

    fea_chunk=${tr_fea_scp_list[$chunk]}
    fea_opts=$tr_fea_opts
    lab_folder=$tr_lab_folder
    lab_opts=$tr_lab_opts

    out_file=$out_folder"/train_ep_"$epoch"_ck_"$chunk".pkl"
    info_file=$out_folder"/train_ep_"$epoch"_ck_"$chunk".info"
    conf_file=$out_folder"/train_ep_"$epoch"_ck_"$chunk".cfg"

    [ -e $conf_file ] && rm $conf_file

    do_training=1
    do_eval=0
    do_forward=0

    # writing config file for training
    write_conf

    # single-chunk training
    if [ ! -f "$info_file" ]; then
     $cmd "${log_dir}/${exp_name}_train.log" python run_nn.py --cfg $conf_file || exit 1
    fi

    # changing random seed for the next chunk
    seed=$(($seed+100))

    # removing previous model
    if [ "$epoch" -gt "0" ]; then
     [ -e $pt_file ] && rm $pt_file
    fi

    pt_file=$out_file

  done

  # Computing total training loss
  loss_tr="$(cat $out_folder"/train_ep_"$epoch"_ck_"*".info" | grep 'loss=' | awk -F "=" '{ sum += $2 } END { if (NR > 0) print sum / NR }')"
  err_tr="$(cat $out_folder"/train_ep_"$epoch"_ck_"*".info" | grep 'err=' | awk -F "=" '{ sum += $2 } END { if (NR > 0) print sum / NR }')"
  time_tr="$(cat $out_folder"/train_ep_"$epoch"_ck_"*".info" | grep 'time=' | awk -F "=" '{ sum += $2 } END { if (NR > 0) print sum}')"

  # eval config
  do_training=0
  do_eval=1
  do_forward=0

  # writing config file for eval
  fea_chunk=$dev_fea_scp
  fea_opts=$dev_fea_opts
  lab_folder=$dev_lab_folder
  lab_opts=$dev_lab_opts

  conf_file=$out_folder"/eval_ep_"$epoch"_ck_"$chunk".cfg"
  out_file=$out_folder"/eval_ep_"$epoch"_ck_"$chunk".info"
  [ -e $conf_file ] && rm $conf_file

  write_conf

  # eval on dev set
  if [ ! -f "$out_file" ]; then
   $cmd "${log_dir}/${exp_name}_eval_dev.log" python run_nn.py --cfg $conf_file || exit 1
  fi

  # Computing total dev loss
  loss_dev="$(cat $out_folder"/eval_ep_"$epoch"_ck_"*".info" | grep 'loss=' | awk -F "=" '{ sum += $2 } END { if (NR > 0) print sum / NR }')"
  err_dev="$(cat $out_folder"/eval_ep_"$epoch"_ck_"*".info" | grep 'err=' | awk -F "=" '{ sum += $2 } END { if (NR > 0) print sum / NR }')"
  time_dev="$(cat $out_folder"/eval_ep_"$epoch"_ck_"*".info" | grep 'time=' | awk -F "=" '{ sum += $2 } END { if (NR > 0) print sum}')"

  # test config
  do_training=0
  do_eval=1
  do_forward=0


  # writing config file for eval
  fea_chunk=$te_fea_scp
  fea_opts=$te_fea_opts
  lab_folder=$te_lab_folder
  lab_opts=$te_lab_opts

  conf_file=$out_folder"/test_ep_"$epoch"_ck_"$chunk".cfg"
  out_file=$out_folder"/test_ep_"$epoch"_ck_"$chunk".info"
  [ -e $conf_file ] && rm $conf_file

  write_conf

  # eval on test set
  if [ ! -f "$out_file" ]; then
   $cmd "${log_dir}/${exp_name}_eval_te.log" python run_nn.py --cfg $conf_file || exit 1
  fi

  # Computing total test loss
  loss_te="$(cat $out_folder"/test_ep_"$epoch"_ck_"*".info" | grep 'loss=' | awk -F "=" '{ sum += $2 } END { if (NR > 0) print sum / NR }')"
  err_te="$(cat $out_folder"/test_ep_"$epoch"_ck_"*".info" | grep 'err=' | awk -F "=" '{ sum += $2 } END { if (NR > 0) print sum / NR }')"
  time_te="$(cat $out_folder"/test_ep_"$epoch"_ck_"*".info" | grep 'time=' | awk -F "=" '{ sum += $2 } END { if (NR > 0) print sum}')"


  printf "epoch %s tr_loss=%s tr_err=%s dev_err=%s test_err=%s learning_rate=%s time=%s sec. \n" $epoch $loss_tr $err_tr $err_dev $err_te $lr $time_tr >>$out_folder/res.res

  # Learning Rate Annealing
  if [ "$epoch" -gt "1" ]; then
   relative_imp=`echo "(($err_dev_prev-$err_dev)/$err_dev)<$improvement_threshold" | bc -l`
  if [ "$relative_imp" -eq "1" ]; then
   lr=`echo "$lr*$halving_factor" | bc -l`
  fi
 fi

done

echo 'Forward...'
# Forward
do_training=0
do_forward=1
do_eval=1  # needed to get N_batches in run_nn.py

fea_chunk=$dev_fea_scp
fea_opts=$dev_fea_opts
lab_folder=$dev_lab_folder
lab_opts=$dev_lab_opts

conf_file=$out_folder"/forward_dev_ep_"$epoch"_ck_"$chunk".cfg"
out_file=$out_folder"/forward_dev_ep_"$epoch"_ck_"$chunk".pkl"
info_file=$out_folder"/forward_dev_ep_"$epoch"_ck_"$chunk".info"

[ -e $conf_file ] && rm $conf_file

write_conf

# generating normalized posteriors for test data
if [ ! -f "$out_file" ]; then
 $cmd "${log_dir}/${exp_name}_forward_dev.log" python run_nn.py --cfg $conf_file || exit 1
fi

fea_chunk=$te_fea_scp
fea_opts=$te_fea_opts
lab_folder=$te_lab_folder
lab_opts=$te_lab_opts

conf_file=$out_folder"/forward_te_ep_"$epoch"_ck_"$chunk".cfg"
out_file=$out_folder"/forward_te_ep_"$epoch"_ck_"$chunk".pkl"
info_file=$out_folder"/forward_te_ep_"$epoch"_ck_"$chunk".info"

[ -e $conf_file ] && rm $conf_file

write_conf

if [ ! -f "$out_file" ]; then
 $cmd "${log_dir}/${exp_name}_forward_te.log" python run_nn.py --cfg $conf_file || exit 1
fi

echo 'Decoding..'
# Decoding
./decode_dnn_TIMIT.sh \
  "${graph_dir}" \
  "${dev_data_folder}" \
  "${tr_lab_folder}" \
  "${out_folder}/decode_dev" \
  "cat ${out_folder}/forward_dev_ep_${epoch}_ck_${chunk}.pkl"

./decode_dnn_TIMIT.sh \
  "${graph_dir}" \
  "${te_data_folder}" \
  "${tr_lab_folder}" \
  "${out_folder}/decode_test" \
  "cat ${out_folder}/forward_te_ep_${epoch}_ck_${chunk}.pkl"
