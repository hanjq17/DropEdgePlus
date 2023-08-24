python ./src/train_new.py     --debug     --datapath data//  \
--dataset pubmed --type mutigcn     --nhiddenlayer 1  \
--nbaseblocklayer 0     --hidden 256    --epoch 400     --lr 0.001  \
--weight_decay 1e-3     --early_stopping 400   \
--dropout 0.8     --normalization BingGeNormAdj     --withbias  \
--b 0.1 --a 0.4 --withloop --withbn