python ./src/train_new.py     --debug     --datapath data//  \
--dataset map --type APPNP     --nhiddenlayer 1  \
--nbaseblocklayer 14     --hidden 64     --epoch 400     --lr 0.005  \
--weight_decay 8e-6     --early_stopping 400    \
--dropout 0.5     --normalization AugNormAdj     --withbias  \
 --b 0.7 --a 0.5 --alpha 0.2