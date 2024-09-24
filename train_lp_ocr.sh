# Total train images: 5666
# 1 epoch = "total_images/batch_size" iterations. 
# 3000 iterations =  50 epochs.
# lmdb_lp_dataset -> ciaws/msil-temp/lpscan/datasets/lmdb_lp_dataset
# lp_ocr_v1: VGG-CTC 6000 iters,  h=100, w=400  
# lp_ocr_v2: VGG-BiLSTM-CTC, 6000iters, h=100, w=400  
# lp_ocr_v3: VGG-BiLSTM-CTC, 6000 iters,  h=150, w=600
# vocab: '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-' # Add this in train.py file.


python train.py \
--result_dir results \
--exp_name lp_ocr_v3 \
--train_data ./lmdb_lp_dataset/train \
--valid_data ./lmdb_lp_dataset/val \
--batch_size 32 \
--select_data "/" \
--batch_ratio '1' \
--num_iter 6000 \
--valInterval 60 \
--augment \
--Transformation None \
--FeatureExtraction VGG \
--SequenceModeling BiLSTM \
--Prediction CTC \
--saved_model "saved_models/None-VGG-BiLSTM-CTC (will be deprecated).pth" \
--batch_max_length 9 \
--imgH 150 \
--imgW 600 \
--FT 