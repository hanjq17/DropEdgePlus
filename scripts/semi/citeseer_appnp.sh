python ./src/train_semi.py     --debug     --datapath data//  \
--dataset citeseer  --type APPNP     --nhiddenlayer 1  \
--nbaseblocklayer 30   --hidden 64     --epoch 400     --lr 0.01  \
--weight_decay 0.0008     --early_stopping 400    \
--dropout 0.5    --normalization NormAdj    --withbias  \
--b 0.9 --a 0.4  --alpha 0.1