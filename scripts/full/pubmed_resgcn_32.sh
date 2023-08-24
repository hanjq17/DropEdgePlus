python ./src/train_new.py     --debug     --datapath data//  \
--dataset pubmed --type resgcn     --nhiddenlayer 1  \
--nbaseblocklayer 30     --hidden 128    --epoch 400     --lr 0.008  \
--weight_decay 0.0008     --early_stopping 400    \
--dropout 0.5     --normalization FirstOrderGCN     --withbias  \
--b 0.7 --a 0.7 --withloop --withbn