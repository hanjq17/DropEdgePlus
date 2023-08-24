python ./src/train_semi.py     --debug     --datapath data// \
--dataset pubmed  --type mutigcn     --nhiddenlayer 1  \
--nbaseblocklayer 2   --hidden 256     --epoch 400     --lr 0.005  \
--weight_decay 0.001     --early_stopping 400   \
--dropout 0.8     --normalization NormAdj     --withbias  \
--b 0.4 --a 0.5
