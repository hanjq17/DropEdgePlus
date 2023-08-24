python ./src/train_semi.py     --debug     --datapath data//  \
--seed 42     --dataset cora  --type APPNP     --nhiddenlayer 1  \
--nbaseblocklayer 6   --hidden 64     --epoch 260     --lr 0.01  \
--weight_decay 0.0008     --early_stopping 400   \
--dropout 0.5    --normalization NormAdj    --withbias  \
--b 0.9 --a 0.8  --alpha 0.1 --loss