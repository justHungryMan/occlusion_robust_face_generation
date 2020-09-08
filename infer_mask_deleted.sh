#python infer_mask_deleted.py --dataset  /data1/Jun/datasets/cctv/new_dataset/mask_delete/test/ \
#--checkpoint /data1/Jun/result/cctv/mask_deleted_model/model_mask_deleted_ep210.ckp \
#--landmark_checkpoint /data1/Jun/result/cctv/pred_landmark/model_landmark_pred_ep1400.ckp \
#--result /data1/Jun/result/cctv/mask_deleted_infer/1/ --gpus 0,1,2,3 --batch_size 32 \
#--result_name mask_deleted

#python infer_mask_deleted.py --dataset  /data1/Jun/datasets/cctv/new_dataset/mySynthesis/ \
#--checkpoint /data1/Jun/result/cctv/mask_deleted_model/model_mask_deleted_ep210.ckp \
#--landmark_checkpoint /data1/Jun/result/cctv/pred_landmark/model_landmark_pred_ep1400.ckp \
#--result /data1/Jun/result/cctv/mask_deleted_infer/2/ --gpus 0,1,2,3 --batch_size 32 \
#--result_name mask_deleted

python infer_mask_deleted.py --dataset  /data1/Jun/datasets/cctv/new_dataset/realworld_mask/ \
--checkpoint /data1/Jun/result/cctv/mask_deleted_model/model_mask_deleted_ep300.ckp \
--landmark_checkpoint /data1/Jun/result/cctv/pred_landmark/model_landmark_pred_ep1400.ckp \
--result /data1/Jun/result/cctv/mask_deleted_infer/4_square/ --gpus 0,1,2,3 --batch_size 32 \
--result_name mask_deleted_square

#python infer_mask_deleted.py --dataset  /data1/Jun/datasets/cctv/new_dataset/sungjun/ \
#--checkpoint /data1/Jun/result/cctv/mask_deleted_model/model_mask_deleted_ep210.ckp \
#--landmark_checkpoint /data1/Jun/result/cctv/pred_landmark/model_landmark_pred_ep1400.ckp \
#--result /data1/Jun/result/cctv/mask_deleted_infer/extra/ --gpus 0,1,2,3 --batch_size 32 \
#--result_name mask_deleted