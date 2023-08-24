python ./src/train_new.py     --debug     --datapath data//  \
 --dataset map --type resgcn     --nhiddenlayer 1  \
--nbaseblocklayer 14     --hidden 256     --epoch 400     --lr 0.006  \
--weight_decay 1e-5     --early_stopping 400    \
--dropout 0.3     --normalization BingGeNormAdj     --withbias  \
--b 0.05 --a 1.0