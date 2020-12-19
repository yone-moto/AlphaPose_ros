alphapose_ros
===============

alphaposeのapiを使ってros化したプログラム

alphaposeのインストール必須。
![参照](https://github.com/MVIG-SJTU/AlphaPose)

なぜかpath が通らなかったのでAlphaPose内以下を編集（2箇所）

darknet.py内　detector/yolo/data/yolov3-spp.weights
darknet.py内　detector/yolo/cfg/yolov3-spp.cfg


できること。
===============
以下参照

以下の２つを起動
roslaunch alphapose_ros alphapose_demo.launch
./alphapose.sh
![muluti_predict](https://user-images.githubusercontent.com/55490093/102687343-ac828e00-4231-11eb-800c-11887f64eb0b.png)
![rosgraph](https://user-images.githubusercontent.com/55490093/102687344-b0161500-4231-11eb-9be2-51c46f8bd9c7.png)
