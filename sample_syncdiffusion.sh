CUDA_VISIBLE_DEVICES=9 python sample_syncdiffusion.py \
--num_samples 1 \
--start 0 \
--seed 100 \
--sync_weight 20.0 \
--sync_freq 1 \
--sync_thres 50 \
--sync_decay_rate 0.95 \
--prompt "natural landscape in anime style illustration" \
--negative "" \
--sd_version "2.0" \
--H 512 \
--W 2048 \
--steps 50 \
--save_dir "results" \
--stride 16

### PROMPTS ###
# natural landscape in anime style illustration
# a photo of a forest with a misty fog
# a photo of a mountain range at twilight
# cartoon panorama of spring summer beautiful nature
# a photo of a snowy mountain peak with skiers
# a photo of a city skyline at night