python ./src/train_new.py     --debug     --datapath data//  \
--dataset mac --type APPNP     --nhiddenlayer 1  \
--nbaseblocklayer 6     --hidden 64     --epoch 400     --lr 0.008  \
--weight_decay 5e-5     --early_stopping 400    \
--dropout 0.5     --normalization AugNormAdj     --withbias  \
 --b 0.4 --a 0.2 --alpha 0.2