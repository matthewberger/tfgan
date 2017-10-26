FILE=$1
URL=http://hdc.cs.arizona.edu/people/berger/tfgan/models/$FILE.tar.gz
TAR_FILE=./models/$FILE.tar.gz
wget -N $URL -O $TAR_FILE
tar xvzf $TAR_FILE -C ./models/
rm $TAR_FILE
