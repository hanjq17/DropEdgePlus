python ./src/train_new.py     --debug     --datapath data//  \
--dataset cora --type APPNP     --nhiddenlayer 1  \
--nbaseblocklayer 30     --hidden 64     --epoch 400     --lr 0.004  \
--weight_decay 5e-5     --early_stopping 400    \
--dropout 0.5     --normalization AugRWalk     --withbias  \
--b 0.5 --a 1.0 --alpha 0.2