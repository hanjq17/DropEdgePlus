python ./src/train_new.py     --debug     --datapath data//  \
--dataset cora --type APPNP     --nhiddenlayer 1  \
--nbaseblocklayer 62     --hidden 128     --epoch 400     --lr 0.006  \
--weight_decay 0.0001     --early_stopping 400    \
--dropout 0.1     --normalization AugNormAdj     --withbias  \
--b 0.5 --a 0.5 --alpha 0.1