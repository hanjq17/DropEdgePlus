python ./src/train_new.py     --debug     --datapath data//  \
--dataset citeseer --type densegcn     --nhiddenlayer 1  \
--nbaseblocklayer 2     --hidden 128     --epoch 400     --lr 0.009  \
--weight_decay 5e-4     --early_stopping 400  \
--dropout 0.8     --normalization BingGeNormAdj     --withbias  \
--b 0.2 --a 0.4 --withloop