python ./src/train_semi.py     --debug     --datapath data// \
--dataset cora --type mutigcn     --nhiddenlayer 1  \
--nbaseblocklayer 2    --hidden 256     --epoch 300     --lr 0.009  \
--weight_decay 0.001     --early_stopping 300   \
--dropout 0.8     --normalization BingGeNormAdj     --withbias  \
--b 0.8 --a 0.2