python ./src/train_new.py     --debug     --datapath data//  \
--dataset pubmed --type resgcn     --nhiddenlayer 1  \
--nbaseblocklayer 6     --hidden 256    --epoch 400     --lr 0.003  \
--weight_decay 1e-3     --early_stopping 400    \
--dropout 0.8     --normalization AugNormAdj     --withbias  \
--b 0.1 --a 0.4 --withloop --withbn