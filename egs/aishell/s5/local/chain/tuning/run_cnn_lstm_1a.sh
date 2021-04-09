#!/usr/bin/env bash
# nnet3-info exp/chain/tdnn_1a_sp/final.mdl
# left-context: 33
# right-context: 15
# num-parameters: 47067170 ...
#
# %WER 5.00 [ 10276 / 205341, 286 ins, 282 del, 9708 sub ] exp/chain/tdnn_1a_sp/decode_dev/cer_7_0.5
# %WER 5.75 [ 6026 / 104765, 140 ins, 298 del, 5588 sub ] exp/chain/tdnn_1a_sp/decode_test/cer_8_1.0
set -e

# configs for 'chain'
affix=
stage=0
train_stage=-10
get_egs_stage=-10
dir=exp/chain/tdnn_1a  # Note: _sp will get added to this
decode_iter=

# training options
num_epochs=4
initial_effective_lrate=0.001
final_effective_lrate=0.0001
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=2
num_jobs_final=12
minibatch_size=64,32
frames_per_eg=150,110,90

remove_egs=false
common_egs_dir=
xent_regularize=0.1
chunk_left_context=40
# decode options
frames_per_chunk_primary=$(echo $frames_per_eg | cut -d, -f1)
extra_left_context=50
extra_right_context=0

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.

dir=${dir}${affix:+_$affix}_sp
train_set=train_sp
ali_dir=exp/tri5a_sp_ali
treedir=exp/chain/tri6_7d_tree_sp
lang=data/lang_chain


# if we are using the speed-perturbed data we need to generate
# alignments for it.
local/chain/run_ivector_common.sh --stage $stage || exit 1;

if [ $stage -le 7 ]; then
  # Get the alignments as lattices (gives the LF-MMI training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat $ali_dir/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/$train_set \
    data/lang exp/tri5a exp/tri5a_sp_lats
  rm exp/tri5a_sp_lats/fsts.*.gz # save space
fi

if [ $stage -le 8 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 9 ]; then
  # Build a tree using our new topology. This is the critically different
  # step compared with other recipes.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 5000 data/$train_set $lang $ali_dir $treedir
fi

if [ $stage -le 10 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)

  cnn_opts="l2-regularize=0.03"
  tdnnf_first_opts="l2-regularize=0.03 dropout-proportion=0.0 bypass-scale=0.0"
  linear_opts="orthonormal-constraint=1.0"
  lstm_opts="l2-regularize=0.0005 decay-time=40"
  lowlrate_opts="learning-rate-factor=0.333 max-change=0.25"
  opts="l2-regularize=0.002"
  ivector_affine_opts="l2-regularize=0.03"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=80 name=input

  idct-layer name=idct input=input dim=80 cepstral-lifter=22 affine-transform-file=$dir/configs/idct.mat
  linear-component name=ivector-linear $ivector_affine_opts dim=240 input=ReplaceIndex(ivector, t, 0) $lowlrate_opts
  batchnorm-component name=ivector-batchnorm target-rms=0.025
  batchnorm-component name=idct-batchnorm input=idct
  spec-augment-layer name=idct-spec-augment freq-max-proportion=0.5 time-zeroed-proportion=0.2 time-mask-max-frames=20
  combine-feature-maps-layer name=combine_inputs input=Append(idct-spec-augment, ivector-batchnorm) num-filters1=1 num-filters2=3 height=80
  # CSPDense block 1
  conv-relu-batchnorm-layer name=cnn1a $cnn_opts height-in=80 height-out=80 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=32 $lowlrate_opts
  combine-feature-maps-layer name=cnn1b input=Append(combine_inputs, cnn1a) num-filters1=4 num-filters2=32 height=80
  conv-relu-batchnorm-layer name=cnn1c $cnn_opts height-in=80 height-out=80 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=32 $lowlrate_opts
  combine-feature-maps-layer name=cnn1d input=Append(cnn1b, cnn1c) num-filters1=36 num-filters2=32 height=80
  conv-relu-batchnorm-layer name=cnn1e $cnn_opts height-in=80 height-out=80 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64 $lowlrate_opts
  combine-feature-maps-layer name=cnn1f input=Append(combine_inputs, cnn1e) num-filters1=4 num-filters2=64 height=80
  conv-relu-batchnorm-layer name=csp1a $cnn_opts height-in=80 height-out=40 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64 $lowlrate_opts
  conv-relu-batchnorm-layer name=csp1b $cnn_opts height-in=80 height-out=40 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64 $lowlrate_opts
  # CSPDense block 2
  conv-relu-batchnorm-layer name=cnn2a input=csp1b $cnn_opts height-in=40 height-out=40 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
  combine-feature-maps-layer name=cnn2b input=Append(csp1b, cnn2a) num-filters1=64 num-filters2=64 height=40
  conv-relu-batchnorm-layer name=cnn2c $cnn_opts height-in=40 height-out=40 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
  combine-feature-maps-layer name=cnn2d input=Append(cnn2b, cnn2c) num-filters1=128 num-filters2=64 height=40
  conv-relu-batchnorm-layer name=cnn2e $cnn_opts height-in=40 height-out=40 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128
  combine-feature-maps-layer name=cnn2f input=Append(csp1a, cnn2e) num-filters1=64 num-filters2=128 height=40
  conv-relu-batchnorm-layer name=csp2a $cnn_opts height-in=40 height-out=20 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128
  conv-relu-batchnorm-layer name=csp2b $cnn_opts height-in=40 height-out=20 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128
  # CSPDense block3
  conv-relu-batchnorm-layer name=cnn3a input=csp2b $cnn_opts height-in=20 height-out=20 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128
  combine-feature-maps-layer name=cnn3b input=Append(csp2b, cnn3a) num-filters1=128 num-filters2=128 height=20
  conv-relu-batchnorm-layer name=cnn3c $cnn_opts height-in=20 height-out=20 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128
  combine-feature-maps-layer name=cnn3d input=Append(cnn3b, cnn3c) num-filters1=256 num-filters2=128 height=20
  conv-relu-batchnorm-layer name=cnn3e $cnn_opts height-in=20 height-out=20 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256
  combine-feature-maps-layer name=cnn3f input=Append(csp2a, cnn3e) num-filters1=128 num-filters2=256 height=20
  conv-relu-batchnorm-layer name=csp3a $cnn_opts height-in=20 height-out=10 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256
  conv-relu-batchnorm-layer name=csp3b $cnn_opts height-in=20 height-out=10 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256
  # CSPLSTMP 1
  linear-component name=lstm1a dim=320 $linear_opts input=Append(-3,0)
  fast-lstmp-layer name=lstm1b cell-dim=1536 recurrent-projection-dim=384 non-recurrent-projection-dim=384 delay=-3 dropout-proportion=0.0 $lstm_opts
  relu-batchnorm-layer name=lstm1c $opts input=Append(0,csp3b) dim=1280
  linear-component name=lstm1d dim=320 $linear_opts input=Append(-3,0,csp3a)
  relu-batchnorm-layer name=csp4a $opts input=Append(lstm1d,csp3a) dim=640
  relu-batchnorm-layer name=csp4b $opts input=Append(lstm1d,csp3a) dim=640
  # CSPLSTMP 2
  linear-component name=lstm2a dim=320 $linear_opts input=Append(-3,0)
  fast-lstmp-layer name=lstm2b cell-dim=1536 recurrent-projection-dim=384 non-recurrent-projection-dim=384 delay=-3 dropout-proportion=0.0 $lstm_opts
  relu-batchnorm-layer name=lstm2c $opts input=Append(0,csp4b) dim=1280
  linear-component name=lstm2d dim=320 $linear_opts input=Append(-3,0,csp4a)
  relu-batchnorm-layer name=csp5a $opts input=Append(lstm2d,csp4a) dim=640
  relu-batchnorm-layer name=csp5b $opts input=Append(lstm2d,csp4a) dim=640
  # CSPLSTM 3
  linear-component name=lstm3a dim=320 $linear_opts input=Append(-3,0)
  fast-lstmp-layer name=lstm3b cell-dim=1536 recurrent-projection-dim=384 non-recurrent-projection-dim=384 delay=-3 dropout-proportion=0.0 $lstm_opts
  relu-batchnorm-layer name=lstm3c $opts input=Append(0,csp5b) dim=1280
  linear-component name=lstm3d dim=320 $linear_opts input=Append(-3,0,csp5a)
  relu-batchnorm-layer name=csp6 $opts input=Append(lstm3d,csp5a) dim=1280

  ## adding the layers for chain branch
  relu-batchnorm-layer name=prefinal-chain input=csp6 dim=625 target-rms=0.5
  output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  relu-batchnorm-layer name=prefinal-xent input=csp6 dim=625 target-rms=0.5
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 11 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/aishell-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir exp/nnet3/ivectors_${train_set} \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_eg \
    --egs.chunk-left-context $chunk_left_context \
    --trainer.add-option="--optimization.memory-compression-level=2 --cuda-memory-proportion=0.9" \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.max-param-change $max_param_change \
    --trainer.deriv-truncate-margin 8 \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/${train_set}_hires \
    --tree-dir $treedir \
    --lat-dir exp/tri5a_sp_lats \
    --dir $dir  || exit 1;
fi

if [ $stage -le 12 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test $dir $dir/graph
fi

graph_dir=$dir/graph
if [ $stage -le 13 ]; then
  for test_set in dev test; do
    steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
      --nj 10 --cmd "$decode_cmd" \
      --extra-left-context $extra_left_context \
      --extra-right-context $extra_right_context \
      --extra-left-context-initial 0 \
      --extra-right-context-final 0 \
      --frames-per-chunk "$frames_per_chunk_primary" \
      --online-ivector-dir exp/nnet3/ivectors_$test_set \
      $graph_dir data/${test_set}_hires $dir/decode_${test_set} || exit 1;
  done
fi

exit;
