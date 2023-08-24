python ./src/train_new.py     --debug     --datapath data//  \
--dataset mac --type densegcn     --nhiddenlayer 1  \
--nbaseblocklayer 6     --hidden 128     --epoch 400     --lr 0.003  \
--weight_decay 5e-5     --early_stopping 400   \
--dropout 0.5     --normalization BingGeNormAdj     --withbias  \
--b 0.8 --a 0.2 --withloop