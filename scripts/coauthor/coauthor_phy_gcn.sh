python ./src/train_new.py     --debug     --datapath data//  \
--dataset map --type mutigcn     --nhiddenlayer 1  \
--nbaseblocklayer 2     --hidden 256     --epoch 400     --lr 0.002  \
--weight_decay 5e-5     --early_stopping 400    \
--dropout 0.1     --normalization AugNormAdj     --withbias  \
 --b 0.3 --a 0.3 --withloop