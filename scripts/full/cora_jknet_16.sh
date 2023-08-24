python ./src/train_semi.py     --debug     --datapath data//  \
--dataset cora --type densegcn     --nhiddenlayer 1  \
--nbaseblocklayer 14    --hidden 256     --epoch 400     --lr 0.001  \
--weight_decay 0.001     --early_stopping 400  \
--dropout 0.8     --normalization AugNormAdj     --withbias  \
--b 0.05 --a 0.1