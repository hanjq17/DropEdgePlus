python ./src/train_new.py     --debug     --datapath data//  \
--dataset aep --type APPNP     --nhiddenlayer 1  \
--nbaseblocklayer 2     --hidden 64     --epoch 400     --lr 0.010  \
--weight_decay 1e-5     --early_stopping 400   \
--dropout 0.1     --normalization AugRWalk     --withbias  \
--b 0.7 --a 0.5 --alpha 0.2