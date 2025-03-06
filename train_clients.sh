#!/bin/bash
#SBATCH --gpus 1
#SBATCH -A Berzelius-2025-50
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user obaidullah.zaland@umu.se

python code_adapt/main_fl.py referit_adapt_afsfpnvisnew5 --ds_to_use='flickr30k' --mdl_to_use='shuffleprune' --bs=2 --resume=False --client 4
python code_adapt/main_fl.py referit_adapt_afsfpnvisnew4 --ds_to_use='flickr30k' --mdl_to_use='shuffleprune' --bs=2 --resume=False --client 3
python code_adapt/main_fl.py referit_adapt_afsfpnvisnew3 --ds_to_use='flickr30k' --mdl_to_use='shuffleprune' --bs=2 --resume=False --client 2
python code_adapt/main_fl.py referit_adapt_afsfpnvisnew2 --ds_to_use='flickr30k' --mdl_to_use='shuffleprune' --bs=2 --resume=False --client 1
python code_adapt/main_fl.py referit_adapt_afsfpnvisnew1 --ds_to_use='flickr30k' --mdl_to_use='shuffleprune' --bs=2 --resume=False --client 0
