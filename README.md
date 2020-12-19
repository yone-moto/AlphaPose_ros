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

<!-- ![rosgraph](http://160.193.150.53:8080/gitbucket/guideDogGroup/alphapose_ros/_attached/1607506758780CfkUirSCkP) -->
