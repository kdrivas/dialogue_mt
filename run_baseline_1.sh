
for corpus in "openSub_enru" "openSub_enes" "openSub_ende" "openSub_enfr"; do
    CUDA_VISIBLE_DEVICES=2,3 fairseq-train \
        data/preprocessed/$corpus \
        --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
        --dropout 0.3 --weight-decay 0.0001 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --max-tokens 4096 \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-detok moses \
        --eval-bleu-remove-bpe \
        --eval-bleu-print-samples \
        --max-epoch 55 \
        --patience 6 \
        --save-dir checkpoint/openSub_ende \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric

    CUDA_VISIBLE_DEVICES=2,3 fairseq-generate data/preprocessed/$corpus \
        --path checkpoint/$corpus/checkpoint_best.pt \
        --batch-size 128 --beam 5 --remove-bpe > "${corpus}_result.txt"
done