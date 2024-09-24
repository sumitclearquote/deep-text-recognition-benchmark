python3 test.py \
--eval_data ./lmdb_lcd_dataset/test \
--batch_size 8 \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--saved_model saved_models/lcd_ocr_v1/best_accuracy.pth --batch_max_length 6 --imgH 120 --imgW 360 \
--character '0123456789.' \
--workers 0