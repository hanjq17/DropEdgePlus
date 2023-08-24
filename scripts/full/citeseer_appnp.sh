python ./src/train_new.py     --debug     --datapath data//  \
--dataset citeseer --type APPNP     --nhiddenlayer 1  \
--nbaseblocklayer 14     --hidden 128     --epoch 400     --lr 0.010  \
--weight_decay 5e-6     --early_stopping 400    \
--dropout 0.5     --normalization AugRWalk     --withbias  \
--b 0.6 --a 0.0 --alpha 0.5