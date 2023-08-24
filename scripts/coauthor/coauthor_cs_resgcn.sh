python ./src/train_new.py     --debug     --datapath data//  \
--dataset mac --type resgcn     --nhiddenlayer 1  \
--nbaseblocklayer 14     --hidden 128     --epoch 400     --lr 0.009  \
--weight_decay 5e-4     --early_stopping 400    \
--dropout 0.1     --normalization AugRWalk     --withbias  \
--b 0.05 --a 0.2 --withloop