from s3_utils import s3_utils
import os

def test_s3_utils():
    # Test for create bucket
    bucket_name = 'test-bucket-by-matsukage'
    s3 = s3_utils(bucket_name)
    assert s3.exist_bucket()
    
    # Test for create directory
    dir_name = 'test-dir-by-matsukage/'
    s3.mk_dir(dir_name)
    assert s3.exist_dir(dir_name)

    # Test for upload file
    test_file_name = 'test.txt'
    with open(test_file_name, 'w') as f:
        f.write(test_file_name)
    s3.upload_file(test_file_name, dir_name)
    assert s3.check_uploaded_file(test_file_name, dir_name)

    # Test for copy file
    new_file_name2 = 'test2.txt'
    s3.copy_file(dir_name + test_file_name, dir_name + new_file_name2)
    assert s3.check_uploaded_file(new_file_name2, dir_name)

    # Test for delete file
    s3.del_file(test_file_name, dir_name)
    assert not s3.check_uploaded_file(test_file_name, dir_name)


    # Test for download file
    if os.path.isfile(new_file_name2): os.remove(new_file_name2)
    s3.download_file(new_file_name2, dir_name)
    assert s3.check_downloaded_file(new_file_name2)

    # Test for delete directory
    s3.del_dir(dir_name)
    assert not s3.exist_dir(dir_name)

    # Test for delete buckets
    s3.del_bucket()
    assert not s3.exist_bucket()

    print('Success')
    os.remove(test_file_name)
    os.remove(new_file_name2)