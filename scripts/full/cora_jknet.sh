python ./src/train_new.py     --debug     --datapath data//  \
--dataset cora --type densegcn     --nhiddenlayer 1  \
--nbaseblocklayer 6     --hidden 256     --epoch 400     --lr 0.006  \
--weight_decay 8e-4     --early_stopping 400    \
--dropout 0.8     --normalization AugRWalk     --withbias  \
--b 0.5 --a 0.6 --withloop