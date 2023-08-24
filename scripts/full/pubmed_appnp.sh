python ./src/train_new.py     --debug     --datapath data//  \
--dataset pubmed --type APPNP     --nhiddenlayer 1  \
--nbaseblocklayer 6     --hidden 128     --epoch 400     --lr 0.008  \
--weight_decay 5e-5     --early_stopping 400    \
--dropout 0.3     --normalization AugRWalk     --withbias  \
--b 0.9 --a 0.6 --alpha 0.5