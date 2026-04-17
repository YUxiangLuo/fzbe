echo "===========BACKEND GIT PULL============"
git pull
sleep 3

echo "===========run actions=============="
./actions.sh

sleep 3

echo "============restart fangzhen-be.service==========="
sudo systemctl restart fangzhen-be.service
