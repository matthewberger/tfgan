FILE=$1
URL=http://hdc.cs.arizona.edu/people/berger/tfgan/datasets/$FILE.tar.gz
TAR_FILE=./datasets/$FILE.tar.gz
wget -N $URL -O $TAR_FILE
tar xvzf $TAR_FILE -C ./datasets/
rm $TAR_FILE
