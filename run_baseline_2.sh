
GPU=3
for corpus in "openSub_ende" "openSub_enfr"; do

    CUDA_VISIBLE_DEVICES=$GPU fairseq-generate data/preprocessed/$corpus \
        --path checkpoint/$corpus/checkpoint_best.pt \
        --batch-size 128 --beam 5 --remove-bpe > "${corpus}_result.txt"
done