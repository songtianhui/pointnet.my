SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

cd $SCRIPTPATH/..
wget https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip --no-check-certificate
unzip modelnet40_ply_hdf5_2048.zip
rm modelnet40_ply_hdf5_2048.zip
cd -
