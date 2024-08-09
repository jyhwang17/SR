for dataset in $1
do
        for model in $2
        do
                for lr in $3
                do
                        for window_length in 50
                        do
                                for dims in 128
                                do
                                        for dropout in 0.5
                                        do
                                                for alpha in 0.0
                                                do
                                                        for mask_prob in $4
                                                        do

                                                                python3 -u sr_train.py \
                                                                --data_path '../../data/SR/' \
                                                                --dataset ${dataset}\
                                                                --model ${model}\
                                                                --mode 'tune' \
                                                                --max_epoch 500\
                                                                --batch_size 256\
                                                                --lr ${lr}\
                                                                --window_length ${window_length}\
                                                                --negs 200\
                                                                --dims ${dims}\
                                                                --alpha ${alpha}\
                                                                --mask_prob ${mask_prob}\
                                                                --dropout ${dropout}
                                                                #>> \ 'log/'${dataset}'/'${model}'_'${window_length}'_'${dims}.log
                                                        done
                                                done
                                        done
                                done
                        done
                done
        done
done
