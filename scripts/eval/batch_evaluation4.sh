srun -p efm_t --job-name=eval_bagel_1gpu --gres=gpu:1 --ntasks-per-node=1 -quotatype=reserved apptainer exec --nv  --bind /mnt:/mnt /mnt/inspurfs/mozi_t/linjingli/apptainer/mllm3r.sif python batch_infer.py \
  --eval-json /mnt/petrelfs/linjingli/UMM_Spatial/annotations/complex_2context_0310_scannet_test.jsonl\
  --image-root /mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data \
  --out-dir /mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/outputs/complex_2context_0310_3data_0310_0030000_complex_2context_0310_scannet_test \
  --model_path /mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/ft_weights/complex_2context_0310_3data_0310_0030000\
