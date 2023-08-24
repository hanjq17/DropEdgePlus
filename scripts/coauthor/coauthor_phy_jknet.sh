python ./src/train_new.py     --debug     --datapath data//  \
  --dataset map --type densegcn     --nhiddenlayer 1  \
--nbaseblocklayer 6     --hidden 256     --epoch 400     --lr 0.008  \
--weight_decay 1e-5     --early_stopping 400     \
--dropout 0.3     --normalization BingGeNormAdj     --withbias  \
 --b 0.7 --a 0.8