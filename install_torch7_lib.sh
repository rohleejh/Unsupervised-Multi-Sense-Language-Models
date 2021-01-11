echo "--install TORCH--" 
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh
sudo apt-get install luarocks
luarocks install mnist && echo "mnist done"
luarocks install luautf8 && echo "lua-utf8 done"
luarocks install cutorch && echo "cutorch"
luarocks install cunn && echo "cunn"
luarocks install manifold && echo "manifold"
luarocks install matio && echo "matio"