python ./src/train_new.py     --debug     --datapath data//  \
--dataset aep --type densegcn     --nhiddenlayer 1  \
--nbaseblocklayer 6     --hidden 256     --epoch 400     --lr 0.010  \
--weight_decay 1e-5     --early_stopping 400   \
--dropout 0.5     --normalization AugRWalk     --withbias  \
--b 0.4 --a 0.3 --withloop