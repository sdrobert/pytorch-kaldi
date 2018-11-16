#! /usr/bin/env bash

. env.sh
. path.sh
. cmd.sh

# options
feats_nj=10
mfcc_conf=conf/feats/mfcc_s5.conf
kaldi_conf=conf/feats/kaldi_41.conf
fbank_json=conf/feats/fbank_41.json
gbank_json=conf/feats/gbank_41.json
tonebank_json=conf/feats/tonebank_41.json
sifbank_json=conf/feats/sifbank_41.json
sigbank_json=conf/feats/sigbank_41.json
sitonebank_json=conf/feats/sitonebank_41.json
s5_dir="${KALDI_ROOT}/egs/timit/s5"
s5_data_dir="${s5_dir}/data"
pybank_conf=conf/feats/pybank.conf
train_chunks=5
dev_chunks=1
test_chunks=1
ali_train_dir="${s5_dir}/exp/tri3_ali"
ali_dev_dir="${s5_dir}/exp/tri3_ali_dev"
ali_test_dir="${s5_dir}/exp/tri3_ali_test"
partial_cfg_folder=conf/partial
data_dir=data
exp_dir=exp
count_file="${s5_dir}/exp/tri3/ali_train_pdf.counts"
gmm_dir="${s5_dir}/exp/tri3"
si_cmvn=false

. utils/parse_options.sh

set -e

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

if [ ! -f "${count_file}" ]; then
  mkdir -p $(dirname "${count_file}")
  if [ ! -f "${ali_train_dir}/final.mdl" ]; then
    echo -e \
"Count file '${count_file}' does not exist and neither does \
'${ali_train_dir}/final.mdl'. Generate the count file yourself."
    exit 1
  fi
  num_pdf=$(hmm-info "${ali_train_dir}/final.mdl" | awk '/pdfs/{print $4}')
  labels_tr_pdf="ark:ali-to-pdf ${ali_train_dir}/final.mdl \"ark:gunzip -c ${ali_train_dir}/ali.*.gz |\" ark:- |"
  analyze-counts \
    --binary=false \
    --counts-dim=$num_pdf \
    "$labels_tr_pdf" "${count_file}"
fi

mkdir -p "${partial_cfg_folder}"
echo "[data]" > "${partial_cfg_folder}/mfcc.cfg"
echo "count_file=\"${count_file}\"" >> "${partial_cfg_folder}/mfcc.cfg"
echo "pt_file=none
" >> "${partial_cfg_folder}/mfcc.cfg"
for x in fmllr kaldi fbank gbank tonebank sifbank sigbank sitonebank ; do
  cp "${partial_cfg_folder}/mfcc.cfg" "${partial_cfg_folder}/$x.cfg"
done
if $si_cmvn ; then
  rm "${partial_cfg_folder}/fmllr.cfg"
fi

for x in train dev test; do
  if [ $x = train ]; then
    num_chunks=${train_chunks}
    ali_dir="${ali_train_dir}"
    cfg_prefix='tr_'
  elif [ $x = dev ]; then
    num_chunks=${dev_chunks}
    ali_dir="${ali_dev_dir}"
    cfg_prefix='dev_'
  else
    num_chunks=${test_chunks}
    ali_dir="${ali_test_dir}"
    cfg_prefix='te_'
  fi
  num_chunks_m1=$(bc -l <<< "${num_chunks} - 1")
  if [ ! -d "${ali_dir}" ]; then
    echo -e "Alignment directory '${ali_dir}' does not exist. Please check \
the README"
    exit 1
  fi

  # get the number of frames for each alignment in this partition
  copy-int-vector \
    "ark:gunzip -c ${ali_dir}/ali.*.gz |" \
    ark,t:- 2>/dev/null | awk '{print $1, NF - 1, ""}' | sort -k1,1 \
    > "${tmpdir}/lens_$x.txt"

  function compare_lengths() {
    # $1 : feature name
    feat-to-len \
      "scp:${s5_data_dir}/$x/feats.scp" \
      "ark,t:-" | sort -k1,1 > "${tmpdir}/comp_$x.txt"
    if ! diff "${tmpdir}/lens_$x.txt" "${tmpdir}/comp_$x.txt" &> /dev/null ; then
      echo -e \
"$1 features don't match the length of the alignments for $x partition. \
Likely, you forgot to add '--snip-edges=false' to the config files when \
# running s5. Consult the README."
      exit 1
    fi
  }

  function append_to_cfg() {
    # $1 : feature name
    fp="${partial_cfg_folder}/$1.cfg"
    echo -n "${cfg_prefix}fea_scp=\"${data_dir}/$1_ord/${x}_split.000" >> "${fp}"
    seq 1 ${num_chunks_m1} | \
      awk '{printf ",'"${data_dir}/$1_ord/${x}_split"'.%0.3d", $1}' >> "${fp}"
    echo '"' >> "${fp}"
    echo "${cfg_prefix}lab_folder=\"${ali_dir}\"" >> "${fp}"
    echo "${cfg_prefix}lab_opts=ali-to-pdf" >> "${fp}"
    echo "${cfg_prefix}data_folder=${data_dir}/$1/$x" >> "${fp}"
    if ${si_cmvn} ; then
      echo -n "${cfg_prefix}fea_opts=\"apply-cmvn ark:${data_dir}/$1_ord/${x}_cmvn_snt.ark ark:- ark:- |" \
        >> "${fp}"
    else
      echo -n "${cfg_prefix}fea_opts=\"apply-cmvn --utt2spk=ark:${data_dir}/$1/utt2spk ark:${data_dir}/$1_ord/${x}_cmvn_speaker.ark ark:- ark:- |" \
        >> "${fp}"
    fi
  }

  cp "${s5_data_dir}/$x/feats.scp" "${s5_data_dir}/$x/feats_old.scp" || true

  # mfcc creation
  steps/make_mfcc.sh \
    --nj ${feats_nj} \
    --mfcc-config "${mfcc_conf}" \
    --cmd "${train_cmd}" \
    --compress false \
    "${s5_data_dir}/$x" "${exp_dir}/log/make_mfcc/$x" "${data_dir}/mfcc/$x"
  compare_lengths mfcc
  mv "${s5_data_dir}/$x/feats.scp" "${data_dir}/mfcc/$x/feats.scp"
  cp "${s5_data_dir}/$x/"{wav.scp,utt2spk,spk2utt,text,stm,glm,spk2gender} \
    "${data_dir}/mfcc/$x"
  cp -r "${s5_data_dir}/$x/split"* "${data_dir}/mfcc/$x/"  # for fmllr
  ./create_chunks.sh \
    "${data_dir}/mfcc/$x" "${data_dir}/mfcc_ord" ${num_chunks} $x 0
  ./create_chunks.sh \
    "${data_dir}/mfcc/$x" "${data_dir}/mfcc_shu" ${num_chunks} $x 1
  append_to_cfg mfcc
  echo " add-deltas --delta-order=2 ark:- ark:- |\"
" >> "${partial_cfg_folder}/mfcc.cfg"

  # mfcc+fmllr creation
  if ! $si_cmvn ; then
    if [ $x = train ]; then
      transform_dir="${ali_train_dir}"
    else
      transform_dir="${gmm_dir}/decode_$x"
    fi
    # we have to temporarily put cmvn in mfcc dir. We'll cut em on exit
    compute-cmvn-stats \
      "--spk2utt=ark:${data_dir}/mfcc/$x/spk2utt" \
      "scp:${data_dir}/mfcc/$x/feats.scp" \
      "ark,scp:${data_dir}/mfcc/$x/cmvn.ark,${data_dir}/mfcc/$x/cmvn.scp"
    trap "rm -f '${data_dir}/mfcc/$x'/cmvn.*" EXIT
    steps/nnet/make_fmllr_feats.sh \
      --nj ${feats_nj} \
      --cmd "${train_cmd}" \
      --transform-dir "${transform_dir}" \
      "${s5_data_dir}/$x" \
      "${data_dir}/mfcc/$x" \
      "${gmm_dir}" \
      "${exp_dir}/log/make_fmllr/$x" \
      "${data_dir}/fmllr/$x"
    compare_lengths fmllr
    mv "${s5_data_dir}/$x/feats.scp" "${data_dir}/fmllr/$x/feats.scp"
    cp "${s5_data_dir}/$x/"{wav.scp,utt2spk,spk2utt,text,stm,glm,spk2gender} \
      "${data_dir}/fmllr/$x"
    ./create_chunks.sh \
      "${data_dir}/mfcc/$x" "${data_dir}/mfcc_ord" ${num_chunks} $x 0
    ./create_chunks.sh \
      "${data_dir}/mfcc/$x" "${data_dir}/mfcc_shu" ${num_chunks} $x 1
    append_to_cfg mfcc
    echo " add-deltas --delta-order=2 ark:- ark:- |\"
" >> "${partial_cfg_folder}/fmllr.cfg"
  fi

  # kaldi (kaldi's fbanks) creation
  steps/make_fbank.sh \
    --nj ${feats_nj} \
    --fbank-config "${kaldi_conf}" \
    --cmd "${train_cmd}" \
    --compress false \
    "${s5_data_dir}/$x" "${exp_dir}/log/make_kaldi/$x" "${data_dir}/kaldi/$x"
  compare_lengths kaldi
  mv "${s5_data_dir}/$x/feats.scp" "${data_dir}/kaldi/$x/feats.scp"
  cp "${s5_data_dir}/$x/"{wav.scp,utt2spk,spk2utt,text,stm,glm,spk2gender} \
    "${data_dir}/kaldi/$x"
  ./create_chunks.sh \
    "${data_dir}/kaldi/$x" "${data_dir}/kaldi_ord" ${num_chunks} $x 0
  ./create_chunks.sh \
    "${data_dir}/kaldi/$x" "${data_dir}/kaldi_shu" ${num_chunks} $x 1
  append_to_cfg kaldi
  echo " add-deltas --delta-order=2 ark:- ark:- |\"
" >> "${partial_cfg_folder}/kaldi.cfg"

  feats=(fbank gbank tonebank sifbank sigbank sitonebank)
  jsons=(
    "${fbank_json}" "${gbank_json}" "${tonebank_json}"
    "${sifbank_json}" "${sigbank_json}" "${sitonebank_json}"
  )
  for idx in $(seq 0 $(echo "${#feats[@]} - 1" | bc)); do
    feat="${feats[$idx]}"
    stepsext/make_pybank.sh \
      --nj ${feats_nj} \
      --compress false \
      --pybank-json "${jsons[$idx]}" \
      --pybank-conf "${pybank_conf}" \
      --cmd "${train_cmd}" \
      "${s5_dir}/${data_dir}/$x" \
      "${exp_dir}/log/make_${feat}/$x" \
      "${data_dir}/${feat}/$x"
    compare_lengths ${feat}
    mv "${s5_data_dir}/$x/feats.scp" "${data_dir}/${feat}/$x/feats.scp"
    cp "${s5_data_dir}/$x/"{wav.scp,utt2spk,spk2utt,text,stm,glm,spk2gender} \
      "${data_dir}/${feat}/$x"
    ./create_chunks.sh \
      "${data_dir}/${feat}/$x" "${data_dir}/${feat}_ord" ${num_chunks} $x 0
    ./create_chunks.sh \
      "${data_dir}/${feat}/$x" "${data_dir}/${feat}_shu" ${num_chunks} $x 1
    append_to_cfg ${feat}
    echo "\"
" >> "${partial_cfg_folder}/${feat}.cfg"
  done

  mv "${s5_data_dir}/$x/feats_old.scp" "${s5_data_dir}/$x/feats.scp" || true
done
