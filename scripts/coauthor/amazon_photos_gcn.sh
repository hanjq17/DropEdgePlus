python ./src/train_new.py     --debug     --datapath data//  \
 --dataset mac --type resgcn     --nhiddenlayer 1  \
--nbaseblocklayer 30     --hidden 64     --epoch 400     --lr 0.009  \
--weight_decay 5e-4     --early_stopping 400    \
--dropout 0.1     --normalization BingGeNormAdj     --withbias  \
 --b 0.2 --a 0.1 --withloop --withbn