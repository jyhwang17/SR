for dataset in $1
do
        for model in 'asmip'
        do
                for lr in $2
                do
                        for window_length in 50
                        do
                                for dims in 128
                                do
                                        for alpha in $3
                                        do
                                                for mask_prob in $4
                                                do
                                                        for shots in 5
                                                        do
                                                            python3 -u task_train.py \
                                                            --data_path '../../data/SR/' \
                                                            --dataset ${dataset}\
                                                            --model ${model}\
                                                            --mode 'tune' \
                                                            --max_epoch 300\
                                                            --batch_size 256\
                                                            --lr ${lr}\
                                                            --window_length ${window_length}\
                                                            --negs 200\
                                                            --dims ${dims}\
                                                            --alpha ${alpha}\
                                                            --mask_prob ${mask_prob}\
                                                            --shots ${shots}\
                                                            --dropout 0.3 >> \
                                                            'log/'${dataset}'/'${model}'_'${window_length}'_'${dims}.log
                                                            
                                                        done
                                                done
                                        done
                                done
                        done
                done
        done
done
