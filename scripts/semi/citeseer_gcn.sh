python ./src/train_semi.py     --debug     --datapath data//   \
--dataset citeseer --type mutigcn     --nhiddenlayer 1  \
--nbaseblocklayer 0    --hidden 128     --epoch 400     --lr 0.009  \
--weight_decay 0.001     --early_stopping 400    \
--dropout 0.8     --normalization BingGeNormAdj     --withbias  \
--b 0.8 --a 0.8