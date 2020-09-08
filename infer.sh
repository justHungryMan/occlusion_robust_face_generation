#python infer.py --dataset  /data1/Jun/datasets/cctv/new_dataset/mySynthesis/ --checkpoint /data1/Jun/result/cctv/model/model_FR_randomMask_newData_angle3_ep420.ckp --result /data1/Jun/result/cctv/randomMask+upAngle3Result_mySynthesis/ --gpus 0,1,2,3 --batch_size 32 --result_name FR_randomMask_upAngle3
#python infer.py --dataset  /data1/Jun/datasets/cctv/new_dataset/DB+upAngle3/test --checkpoint /data1/Jun/result/cctv/model/model_FR_randomMask_newData_angle3_ep420.ckp --result /data1/Jun/result/cctv/randomMask+upAngle3Result/ --gpus 0,1,2,3 --batch_size 32 --result_name FR_randomMask_upAngle3

#python infer.py --dataset  /data1/Jun/datasets/cctv/new_dataset/cctvView --checkpoint /data1/Jun/result/cctv/model/model_FR_randomMask_newData_angle3_ep420.ckp --result /data1/Jun/result/cctv/cctvView/ --gpus 0,1,2,3 --batch_size 32 --result_name FR_randomMask_upAngle3
python infer.py --dataset  /data1/Jun/datasets/cctv/new_dataset/sungjun --checkpoint /data1/Jun/result/cctv/model/model_FR_randomMask_newData_angle3_ep420.ckp --result /data1/Jun/result/cctv/sungjun/ --gpus 0,1,2,3 --batch_size 32 --result_name FR_randomMask_upAngle3

