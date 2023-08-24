python ./src/train_semi.py     --debug     --datapath data// \
--dataset cora --type resgcn     --nhiddenlayer 1  \
--nbaseblocklayer 6    --hidden 64     --epoch 300     --lr 0.007  \
--weight_decay 5e-5     --early_stopping 300    \
--dropout 0.8     --normalization AugRWalk     --withbias  \
--b 0.6 --a 1