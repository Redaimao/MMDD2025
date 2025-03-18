for modality in "va" "af" "vf" "vaf"
do
  for type in "concat" "transformer" "senet"
  do
    CUDA_VISIBLE_DEVICES=1 python train_test_feature.py --train_dataset "MU3D" \
    --train_list '/pathtofeature/MU3D/MU3D_features.pkl' \
    --test_dataset "BagOfLies" \
    --test_list '/pathtofeature/BagOfLies/BgOL_test_features.pkl' \
    --fusion_type $type \
    --modalities $modality
  done
done





