#Training with shufflenet
python code_adapt/main_dist.py referit_adapt_afsfpnvisnew --ds_to_use='refclef' --mdl_to_use='shuffleprune' --bs=1 --resume=False

# Training with Resnet
python code_adapt/main_dist.py referit_adapt_afsfpnvisnew --ds_to_use='refclef' --mdl_to_use='retinaprune' --bs=1 --resume=False


#Testing
#python code_adapt/main_dist.py referit_adapt_afsfpnvisnew --ds_to_use='refclef' --mdl_to_use='shuffleprune' --bs=1 --resume=True --resume_path='tmp_wi_garan_flickr30k_att_loss/models/referit_adapt_afsfpnvisnew.pth' --only_test=True
