# Total images: 1850. 1 epoch = "total_images/batch_size" iterations. 
# 3000 iterations =  50 epochs.

# lcd_ocr_v1: 50 epochs, h=120, w=360    


python3 train.py \
--exp_name lcd_ocr_v1     \
--train_data ./lmdb_lcd_dataset/train \
--valid_data ./lmdb_lcd_dataset/val \
--batch_size 32 \
--select_data "/" \
--batch_ratio '1' \
--num_iter 3000 \ 
--valInterval 58 \
--augment \
--log_wandb \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--saved_model saved_models/TPS-ResNet-BiLSTM-Attn.pth \
--batch_max_length 6 \
--imgH 120 \
--imgW 360 \
--FT 