python train_deleted_gt_landmark.py --dataset /data1/Jun/datasets/cctv/new_dataset/mask_delete/train/\
 --save_model /data1/Jun/result/cctv/mask_deleted_model/ --training_result /data1/Jun/result/cctv/mask_deleted_result/\
  --epochs 5000 --gpus 0,1,2,3 --batch_size 64 --num_workers 16 --result_name mask_deleted --loss ls