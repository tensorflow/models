This directory contains the data needed for training and benchmarking various
navigation models.

1.  Download the data from the [dataset website]
    (http://buildingparser.stanford.edu/dataset.html).
    1.  [Raw meshes](https://goo.gl/forms/2YSPaO2UKmn5Td5m2). We need the meshes
        which are in the noXYZ folder. Download the tar files and place them in
        the `stanford_building_parser_dataset_raw` folder. You need to download
        `area_1_noXYZ.tar`, `area_3_noXYZ.tar`, `area_5a_noXYZ.tar`,
        `area_5b_noXYZ.tar`, `area_6_noXYZ.tar` for training and
        `area_4_noXYZ.tar` for evaluation.
    2.  [Annotations](https://goo.gl/forms/4SoGp4KtH1jfRqEj2) for setting up
        tasks. We will need the file called `Stanford3dDataset_v1.2.zip`. Place
        the file in the directory `stanford_building_parser_dataset_raw`.

2.  Preprocess the data.
    1.  Extract meshes using `scripts/script_preprocess_meshes_S3DIS.sh`. After
        this `ls data/stanford_building_parser_dataset/mesh` should have 6
        folders `area1`, `area3`, `area4`, `area5a`, `area5b`, `area6`, with
        textures and obj files within each directory.
    2.  Extract out room information and semantics from zip file using
        `scripts/script_preprocess_annoations_S3DIS.sh`. After this there should
        be `room-dimension` and `class-maps` folder in
        `data/stanford_building_parser_dataset`. (If you find this script to
        crash because of an exception in np.loadtxt while processing
        `Area_5/office_19/Annotations/ceiling_1.txt`, there is a special
        character on line 323474, that should be removed manually.)

3.  Download ImageNet Pre-trained models. We used ResNet-v2-50 for representing
    images. For RGB images this is pre-trained on ImageNet. For Depth images we
    [distill](https://arxiv.org/abs/1507.00448) the RGB model to depth images
    using paired RGB-D images. Both there models are available through
    `scripts/script_download_init_models.sh`
