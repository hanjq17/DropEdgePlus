python ./src/train_new.py     --debug     --datapath data//  \
--dataset mac --type mutigcn     --nhiddenlayer 1  \
--nbaseblocklayer 0     --hidden 128     --epoch 400     --lr 0.006  \
--weight_decay 1e-3     --early_stopping 400   \
--dropout 0.5     --normalization AugRWalk     --withbias  \
 --b 0.2 --a 0.7 --withloop