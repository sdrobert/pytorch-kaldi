#! /usr/bin/env bash
# Run a series of experiments

. path.sh
. cmd.sh

exp_dir=exp
log_dir=exp/log
num_trials=5
ctype=ord
help_message="Usage: $0 [options] <factor1> [<factor2> ...]"
# these configs will be included, but not as factors, nor will they be named
unlabeled_configs=

. utils/parse_options.sh

if [ $# -lt 1 ]; then
  echo -e "$help_message"
  exit 1
fi

set -e

# warning! this'll prepend a '_' to every name. We remove it later
exp_names=( '' )
exp_args=( '' )

while [ $# != 0 ]; do
  if [ -z "$1" ]; then
    echo -e "Factor must not be empty"
    exit 1
  fi
  IFS=, read -a factor_values <<< $1
  new_exp_names=()
  new_exp_args=()
  # we change last factors fastest
  for exp_name in "${exp_names[@]}"; do
    for factor_value in "${factor_values[@]}"; do
      if [ ! -z "${factor_value}" ]; then
        new_exp_names+=( "${exp_name}_${factor_value}" )
      else
        # assume an empty factor value just means use the defaults. At which
        # point, do not
        new_exp_names+=( "${exp_name}" )
      fi
    done
  done
  for exp_arg in "${exp_args[@]}"; do
    for factor_value in "${factor_values[@]}"; do
      new_exp_args+=( "${exp_arg} ${factor_value}" )
    done
  done
  exp_names=( "${new_exp_names[@]}" )
  exp_args=( "${new_exp_args[@]}" )
  shift
done

for i in $(seq 0 $(echo "${#exp_names[@]} - 1" | bc)); do
  exp_name="${exp_names[$i]:1}"
  exp_arg="${exp_args[$i]}"
  for trial in $(seq 1 ${num_trials}); do
    echo "Beginning trial ${trial} with configs ${exp_arg}"
    run_one_exp.sh \
      --seed "${trial}" \
      --ctype "${ctype}" \
      --log-dir "${log_dir}/${exp_name}_${trial}" \
      "${exp_dir}/${exp_name}_${trial}" ${exp_arg} ${unlabeled_configs//,/ }
    echo "Done trial ${trial} with configs "${exp_arg}""
  done
done
