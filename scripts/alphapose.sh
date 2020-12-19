rosrun alphapose_ros Alphapose_ros.py\
    --cfg /home/robog/AlphaPose/configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml \
    --checkpoint /home/robog/AlphaPose/pretrained_models/halpe26_fast_res50_256x192.pth \
    --vis \
    --detector tracker \
    --pose_track \
    --vis_fast \
    # --profile \
    

    # --showbox 
    # --pose_track \
    # --detbatch 1