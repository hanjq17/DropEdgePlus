python ./src/train_semi.py     --debug     --datapath data//  \
--dataset citeseer --type resgcn     --nhiddenlayer 1  \
--nbaseblocklayer 62    --hidden 64     --epoch 400     --lr 0.007  \
--weight_decay 0.001     --early_stopping 400   \
--dropout 0.8     --normalization AugRWalk     --withbias  \
--b 0.6 --a 0.6