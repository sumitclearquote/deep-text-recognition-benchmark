# 1 epoch = "total_images/batch_size" iterations.

# Iterations:-----------------
#1.  odo_ocr_vgg_ctc_v1
#       Total images:train: 7768, val: 1242
#       h = 40, w= 100
#       vocab: "0123456789."
#       pretrained model: None-VGG-None-CTC.pth
#       num_iter: 3000 (7768/256 = 30. 1 epoch = 30 iters => 100 epochs = 3000 iters)



python train.py \
--result_dir = results/odo_ocr \
--exp_name odo_ocr_vgg_ctc_v1 \
--train_data ./lmdb_odo_ocr_dataset/train \
--valid_data ./lmdb_odo_ocr_dataset/val \
--batch_size 256 \
--select_data "/" \
--batch_ratio '1' \
--num_iter 3000 \
--valInterval 60 \
--augment \
--log_wandb \
--Transformation None \
--FeatureExtraction VGG \
--SequenceModeling None \
--Prediction CTC \
--saved_model "saved_models/None-VGG-None-CTC.pth"h \
--batch_max_length 9 \
--imgH 40 \
--imgW 100 \
--FT 