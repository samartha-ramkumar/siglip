# rm -r "src_tf/data/flickr8k/"
mkdir -p src_tf/data/flickr8k/

if [ ! -f "src_tf/data/flickr8k.zip" ]
then
    wget "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr8k.zip" -O "src_tf/data/flickr8k.zip"
fi

if [ ! -d "src_tf/data/flickr8k/" ]
then
    unzip "src_tf/data/flickr8k.zip" -d src_tf/data/flickr8k/
    rm src_tf/data/flickr8k.zip
    echo "Downloaded Flickr8k dataset successfully."
fi