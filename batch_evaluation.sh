python batch_infer.py \
  --eval-json /mnt/petrelfs/linjingli/UMM_Spatial/annotations/refine_step_prompt_scannet_test.jsonl\
  --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
  --out-dir /mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/outputs/step_mix_refine0227_0015000_refine_step_prompt_scannet_test \
  --model_path /mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/ft_weights/step_mix_refine0227_0015000 \
  --add_context_role_text

python batch_infer.py \
  --eval-json /mnt/petrelfs/linjingli/UMM_Spatial/annotations/refine_step_prompt_dl3dv_test.jsonl\
  --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
  --out-dir /mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/outputs/step_mix_refine0227_0015000_refine_step_prompt_dl3dv_test \
  --model_path /mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/ft_weights/step_mix_refine0227_0015000 \
  --add_context_role_text

python batch_infer.py \
  --eval-json /mnt/petrelfs/linjingli/UMM_Spatial/annotations/refine_step_prompt_matterport3d_test.jsonl\
  --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
  --out-dir /mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/outputs/step_mix_refine0227_0015000_refine_step_prompt_matterport3d_test \
  --model_path /mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/ft_weights/step_mix_refine0227_0015000 \
  --add_context_role_text

python batch_infer.py \
  --eval-json /mnt/petrelfs/linjingli/UMM_Spatial/annotations/step_prompt_scannet_test.jsonl\
  --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
  --out-dir /mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/outputs/step_mix0227_0015000_step_prompt_scannet_test \
  --model_path /mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/ft_weights/step_mixe0227_0015000 \
  --add_context_role_text

python batch_infer.py \
  --eval-json /mnt/petrelfs/linjingli/UMM_Spatial/annotations/step_prompt_dl3dv_test.jsonl\
  --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
  --out-dir /mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/outputs/step_mix0227_0015000_step_prompt_dl3dv_test \
  --model_path /mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/ft_weights/step_mix0227_0015000 \
  --add_context_role_text

python batch_infer.py \
  --eval-json /mnt/petrelfs/linjingli/UMM_Spatial/annotations/step_prompt_matterport3d_test.jsonl\
  --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
  --out-dir /mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/outputs/step_mix0227_0015000_step_prompt_matterport3d_test \
  --model_path /mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/ft_weights/step_mix0227_0015000 \
  --add_context_role_text