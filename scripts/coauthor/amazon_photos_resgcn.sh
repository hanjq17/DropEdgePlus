python ./src/train_new.py     --debug     --datapath data//  \
--dataset aep --type resgcn     --nhiddenlayer 1  \
--nbaseblocklayer 2     --hidden 256     --epoch 400     --lr 0.006  \
--weight_decay 1e-4     --early_stopping 400  \
--dropout 0.8     --normalization FirstOrderGCN     --withbias  \
--b 0.3 --a 0.8 --withloop