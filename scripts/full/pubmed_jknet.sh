python ./src/train_new.py     --debug     --datapath data//  \
--dataset pubmed --type densegcn     --nhiddenlayer 1  \
--nbaseblocklayer 6     --hidden 128    --epoch 400     --lr 0.005  \
--weight_decay 1e-4     --early_stopping 400     \
--dropout 0.8     --normalization AugRWalk     --withbias  \
--b 0.6 --a 0.9 --withloop --withbn