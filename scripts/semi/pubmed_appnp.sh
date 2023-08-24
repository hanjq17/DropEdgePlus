python ./src/train_semi.py     --debug     --datapath data//  \
--dataset pubmed  --type APPNP     --nhiddenlayer 1  \
--nbaseblocklayer 62   --hidden 256     --epoch 100    --lr 0.004  \
--weight_decay 0.0001     --early_stopping 400     \
--dropout 0.1    --normalization BingGeNormAdj    --withbias  \
--b 0.3 --a 0.2  --alpha 0.5
