from main import main
from s3.s3_utils import s3_utils
import os, sys

sys.path.append('/home/runner/work/BboxSuggestion/BboxSuggestion/models')
sys.path.append('/home/runner/work/BboxSuggestion/BboxSuggestion/models/research')

def test_main():
    user_id:str = 'sample-id'
    s3 = s3_utils('bounding-box-suggestion')

    # clear output on s3
    if s3.check_uploaded_file('output.zip', f'{user_id}/output/'): s3.del_file('output.zip', f'{user_id}/output/')

    # inference
    main(user_id)

    # clear output on local
    assert os.path.isfile('output.zip')
    os.system('rm -rf output.zip')
    assert not os.path.isfile('output.zip')

    assert os.path.isdir('output')
    os.system('rm -rf output')
    assert not os.path.isdir('output')

    # download output from s3
    assert s3.check_uploaded_file('output.zip', f'{user_id}/output/')
    s3.download_file('output.zip', f'{user_id}/output/')
    assert os.path.isfile('output.zip')

    # unzip
    os.system('unzip -o output.zip')
    assert os.path.isdir('output')
    assert os.path.isfile('output/human.jpeg')
    assert os.path.isfile('output/dog.jpeg')