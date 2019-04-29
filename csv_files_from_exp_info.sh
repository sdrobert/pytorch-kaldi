#! /usr/bin/env bash

# print out a csv file containing loss and error info over the course of an
# experiment

. env.sh
. path.sh

if [ $# = 0 ]; then
  echo -e "Usage: $0 <exp-dir-1> <exp-dir-2> ..."
  echo -e "e.g. $0 exp/liGRU_mfcc_{1,2,3,4,5}"
  exit 1
fi

tmpdir="$(mktemp -d)"
trap "rm -rf '${tmpdir}'" EXIT

train_chunks=
eval_chunks=
for f in "$@"; do
  cur_train_chunks=( $(ls "${f}/train_ep_01_ck_"*".info" | awk -F'[_.]' '{print $(NF - 1)}') )
  cur_eval_chunks=( $(ls "${f}/eval_ep_01_ck_"*".info" | awk -F'[_.]' '{print $(NF - 1)}') )
  if [ -z "${train_chunks}" ]; then
    train_chunks=( "${cur_train_chunks[@]}" )
  elif [ "${train_chunks[*]}" != "${cur_train_chunks[*]}" ]; then
    echo -e "Experiments have different training chunks"
    exit 1
  fi
  if [ -z "${eval_chunks}" ]; then
    eval_chunks=( "${cur_eval_chunks[@]}" )
  elif [ "${eval_chunks[*]}" != "${cur_eval_chunks[*]}" ]; then
    echo -e "Experiments have different evaluation chunks"
    exit 1
  fi
done

echo -n "iter,epoch"
for ck in "${train_chunks[@]}"; do
  echo -n ",train_ck_${ck}_loss,train_ck_${ck}_err"
done
for ck in "${eval_chunks[@]}"; do
  echo -n ",eval_ck_${ck}_loss,eval_ck_${ck}_err"
done
echo ",lr"

it=0
for f in "$@"; do
  it=$(bc -l <<< "${it} + 1")
  epochs=( $(ls "${f}/eval_ep_"*"_ck_${eval_chunks}.info" | awk -F'_' '{print $(NF - 2)}') )
  for epoch in "${epochs[@]}"; do
    echo -n "${it},${epoch}"
    for ck in "${train_chunks[@]}"; do
      awk -F'=' '$1 == "loss" || $1 == "err" {printf ",%s", $2}' "${f}/train_ep_${epoch}_ck_${ck}.info"
    done
    for ck in "${eval_chunks[@]}"; do
      awk -F'=' '$1 == "loss" || $1 == "err" {printf ",%s", $2}' "${f}/eval_ep_${epoch}_ck_${ck}.info"
    done
    awk -F'=' '$1 == "lr" {printf ",%f", $2}' "${f}/eval_ep_${epoch}_ck_${ck}.cfg"
    echo ""
  done
done
