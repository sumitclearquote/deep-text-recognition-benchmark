# Total train images: 5666
# 1 epoch = "total_images/batch_size" iterations. 
# 3000 iterations =  50 epochs.
# lmdb_lp_dataset -> ciaws/msil-temp/lpscan/datasets/lmdb_lp_dataset
# lp_ocr_v1:  epochs, h=100, w=400  
# vocab: '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-' # Add this in train.py file.


python train.py \
--result_dir results \
--exp_name lp_ocr_v1 \
--train_data ./lmdb_lp_dataset/train \
--valid_data ./lmdb_lp_dataset/val \
--batch_size 32 \
--select_data "/" \
--batch_ratio '1' \
--num_iter 3000 \
--valInterval 60 \
--augment \
--Transformation None \
--FeatureExtraction VGG \
--SequenceModeling None \
--Prediction CTC \
--saved_model "saved_models/None-VGG-None-CTC.pth" \
--batch_max_length 9 \
--imgH 100 \
--imgW 400 \
--FT 