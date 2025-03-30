mkdir -p data/flickr8k/
# rm -r "data/flickr8k/"

if [ ! -f "data/flickr8k.zip" ]
then
    wget "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr8k.zip" -O "data/flickr8k.zip"
fi

if [ ! -d "data/flickr8k/" ]
then
    unzip "data/flickr8k.zip" -d ./data/flickr8k/
    rm data/flickr8k.zip
    echo "Downloaded Flickr8k dataset successfully."
fi