python ./src/train_new.py     --debug     --datapath data//  \
--dataset citeseer --type mutigcn     --nhiddenlayer 1  \
--nbaseblocklayer 0     --hidden 256     --epoch 400     --lr 0.006  \
--weight_decay 1e-3     --early_stopping 400   \
--dropout 0.5     --normalization AugNormAdj     --withbias  \
--b 0.4 --a 0.9 --withloop