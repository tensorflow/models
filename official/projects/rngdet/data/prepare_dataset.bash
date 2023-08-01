# Download sat2graph dataset
# # This link is invalid now: wget https://mapster.csail.mit.edu/sat2graph/data.zip
# if you cannot download this by script, download it manually at https://drive.google.com/file/d/1R8sI1RmFe3rUfWMQaOfsYlBDHpQxFH-H/view?usp=share_link
gdown https://drive.google.com/uc?id=1R8sI1RmFe3rUfWMQaOfsYlBDHpQxFH-H
unzip data.zip
rm -rf data.zip 
mkdir -p ./dataset
mv ./data/* ./dataset

# Generate label
echo "Generating labels ..."
python create_label.py
python data_split.py
echo "Finsh generating labels!"