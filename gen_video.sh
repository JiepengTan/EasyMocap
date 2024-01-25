
# 加速梯子
source /etc/network_turbo

# 设置变量
data=./data/examples/internet-rotate
openposedir=./openpose
videoname=23EfsN7vEOA+003170+003670

echo "起始时间：$(date "+%H:%M:%S")"
# 删除缓存
rm -rf ./output
rm -rf ${data}/images
rm -rf ${data}/annots
rm -rf ${data}/openpose
rm -rf ./output.zip

# hrnet 生成基本keypoint
python3 apps/preprocess/extract_keypoints.py ${data} --mode yolo-hrnet 

# openpose 增加手和脚 keypoint 
python3 apps/preprocess/extract_keypoints.py ${data} --mode feetcrop --openpose ./openpose --hand --force

# 使用生成好的keypoint 生成3D mesh 并 进一步进行降噪音 
PYOPENGL_PLATFORM=osmesa emc --data config/datasets/svimage.yml --exp config/1v1p/hrnet_pare_finetune.yml --root ${data} --ranges 0 500 1 --subs ${videoname}

echo "结束时间：$(date "+%H:%M:%S")"
zip -r ouput.zip ./output > /dev/null 2>&1



