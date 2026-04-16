echo "===========BACKEND GIT PULL============"
git pull
sleep 1
sudo systemctl restart fangzhen-be.service
