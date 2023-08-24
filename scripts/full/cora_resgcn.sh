python ./src/train_new.py     --debug     --datapath data//  \
--dataset cora --type resgcn     --nhiddenlayer 1  \
--nbaseblocklayer 2     --hidden 256     --epoch 400     --lr 0.004  \
--weight_decay 5e-3     --early_stopping 400  \
--dropout 0.1     --normalization AugRWalk     --withbias  \
--b 0.1 --a 0.3 --withloop