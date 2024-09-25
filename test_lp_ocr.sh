python3 test.py \
--eval_data ./datasets/lmdb_lp_dataset_uk/test \
--batch_size 8 \
--Transformation None \
--FeatureExtraction VGG \
--SequenceModeling None \
--Prediction CTC \
--saved_model results/lp_ocr_v3/best_accuracy.pth \
--batch_max_length 7 \
--imgH 100 \
--imgW 400 \
--character '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ' \
--workers 0