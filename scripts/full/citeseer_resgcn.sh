python ./src/train_new.py     --debug     --datapath data//  \
--dataset citeseer --type resgcn     --nhiddenlayer 1  \
--nbaseblocklayer 14     --hidden 256     --epoch 400     --lr 0.010  \
--weight_decay 5e-5     --early_stopping 400     \
--dropout 0.3     --normalization BingGeNormAdj     --withbias  \
--b 0.5 --a 0.7 --withloop