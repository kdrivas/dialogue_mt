#!/usr/bin/env bash

#git clone https://github.com/moses-smt/mosesdecoder.git
#git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=10000

prep=data/inter/agnostic/pass
tmp=$prep/tmp

mkdir -p $tmp
for corpus in "openSub_ende" "openSub_enfr" "openSub_enru" "openSub_enes"; do
    orig=data/inter/agnostic/$corpus
    src=${corpus: -4:-2}
    tgt=${corpus: -2}   
    
    echo "pre-processing train, test and dev data..."
    for l in $src $tgt; do
        for partition in "train" "dev" "test"; do
            f=$partition.$l

            cat $orig/$f | \
                perl $NORM_PUNC $l | \
                perl $TOKENIZER -threads 8 -l $l > $tmp/$f
            echo ""
        done
    done

    TRAIN=$tmp/train.all
    BPE_CODE=$prep/code
    rm -f $TRAIN
    for l in $src $tgt; do
        cat $tmp/train.$l >> $TRAIN
    done

    echo "learn_bpe.py on ${TRAIN}..."
    python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

    for L in $src $tgt; do
        for f in train.$L dev.$L test.$L; do
            echo "apply_bpe.py to ${f}..."
            python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
        done
    done

    TEXT=data/inter/agnostic/pass
    fairseq-preprocess --source-lang $src --target-lang $tgt \
        --trainpref $TEXT/train --validpref $TEXT/dev --testpref $TEXT/test \
        --destdir data/preprocessed/$corpus \
        --workers 20
done
    
#CUDA_VISIBLE_DEVICES=2 fairseq-train \
#    data/preprocessed/openSub_ende \
#    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
#    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#    --dropout 0.3 --weight-decay 0.0001 \
#    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#    --max-tokens 4096 \
#    --eval-bleu \
#    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#    --eval-bleu-detok moses \
#    --eval-bleu-remove-bpe \
#    --eval-bleu-print-samples \
#    --max-epoch 30 \
#    --patience 5 \
#    --save-dir checkpoint/openSub_ende \
#    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
    
#CUDA_VISIBLE_DEVICES=2 fairseq-generate data/preprocessed/openSub_ende \
#    --path checkpoint/checkpoint_best.pt \
#    --batch-size 128 --beam 5 --remove-bpe