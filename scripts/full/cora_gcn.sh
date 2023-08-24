python ./src/train_new.py     --debug     --datapath data//  \
--dataset cora --type mutigcn     --nhiddenlayer 1  \
--nbaseblocklayer 0     --hidden 256     --epoch 400     --lr 0.004  \
--weight_decay 5e-4     --early_stopping 400  \
--dropout 0.3     --normalization AugRWalk     --withbias  \
--b 0.1 --a 0.4