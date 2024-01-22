import boto3, os

# s3リソースへアクセスするためのクラス
class s3_utils:
    s3_resource = boto3.resource('s3', region_name="ap-northeast-1")
    s3_clinet = boto3.client('s3')
    bucket_name=""

    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        if not self.exist_bucket():
            self.mk_bucket(self.bucket_name)

    # Check if the bucket on S3 exists
    def exist_bucket(self):
        name_list = [b.name for b in self.s3_resource.buckets.all()]
        if self.bucket_name in name_list:
            return True
        else:
            return False
    
    # Check if the directory on S3 exists
    def exist_dir(self, dir):
        from botocore.errorfactory import ClientError
        if not dir.endswith('/'): dir = dir + '/'
        try:
            self.s3_clinet.head_object(Bucket=self.bucket_name, Key=dir)
            return True
        except ClientError:
            return False

    # Create a bucket on S3
    def mk_bucket(self, bucket_name):
        bucket = self.s3_resource.Bucket(bucket_name)
        bucket.create(
            CreateBucketConfiguration={
                'LocationConstraint': 'ap-northeast-1'
            }
        )
        assert self.exist_bucket()
        print(f'Created {bucket_name}')

    # Create a directory on S3
    def mk_dir(self, dir):
        if not self.exist_bucket():
            print('Not exitst bucket')
            assert False
        if not self.exist_dir(dir):
            self.s3_clinet.put_object(Bucket=self.bucket_name, Key=dir)
        assert self.exist_dir(dir)
        print(f'Created {dir}')

    # Delete a directory on S3
    def del_dir(self, dir):
        if not self.exist_bucket(): return
        if self.exist_dir(dir):
            bucket = self.s3_resource.Bucket(self.bucket_name)
            if not dir.endswith('/'): dir += '/'
            bucket.objects.filter(Prefix=dir).delete()
        assert not self.exist_dir(dir)
        print(f'Deleted {dir}')

    # Delete all objects in the bucket and the bucket itself
    def del_bucket(self):
        if not self.exist_bucket(): return
        bucket = self.s3_resource.Bucket(self.bucket_name)
        bucket.object_versions.delete()
        self.s3_clinet.delete_bucket(Bucket=self.bucket_name)
        assert not self.exist_bucket()
        print(f'Deleted {self.bucket_name}')
    
    # Upload a file to S3
    def upload_file(self, file_name, dir=''):
        if not self.exist_bucket():
            print('Not exitst bucket')
            assert False
        if not self.exist_dir(dir):
            self.mk_dir(dir)
        fname = os.path.basename(file_name)
        self.s3_resource.Object(self.bucket_name, dir+fname).upload_file(file_name)
        assert self.check_uploaded_file(file_name, dir)
        print(f'Uploaded {file_name} to {dir}')

    
    # Check uploaded file
    def check_uploaded_file(self, file_name, dir=''):
        if not self.exist_bucket():
            print('Not exitst bucket')
            assert False
        bucket = self.s3_resource.Bucket(self.bucket_name)
        if not dir.endswith('/'): dir += '/'
        for obj in bucket.objects.filter(Prefix=dir + file_name):
            return True
        return False
    
    # Download a file from S3
    def download_file(self, file_name, dir=''):
        if not self.exist_bucket():
            print('Not exitst bucket')
            assert False
        if not self.exist_dir(dir):
            print('Not exitst dir')
            assert False
        self.s3_resource.Object(self.bucket_name, dir+file_name).download_file(file_name)
        assert self.check_downloaded_file(file_name)
        print(f'Downloaded {file_name} from {dir}')
    
    # Check downloaded file
    def check_downloaded_file(self, file_name):
        return os.path.isfile(file_name)
    
    # Delete a file on S3
    def del_file(self, file_name, dir=''):
        if not self.exist_bucket():
            print('Not exitst bucket')
            assert False
        if not self.exist_dir(dir):
            print('Not exitst dir')
            assert False
        self.s3_clinet.delete_object(Bucket=self.bucket_name, Key=dir+file_name)
        assert not self.check_uploaded_file(file_name, dir)
        print(f'Deleted {file_name} from {dir}')
    
    # Copy a file on S3
    def copy_file(self, src_file_key, new_file_key):
        if not self.exist_bucket():
            print('Not exitst bucket')
            assert False
        self.s3_clinet.copy_object(Bucket=self.bucket_name, CopySource=self.bucket_name+'/'+src_file_key, Key=new_file_key)
        new_file_name = new_file_key.split('/')[-1]
        dir = '/'.join(new_file_key.split('/')[:-1])
        assert self.check_uploaded_file(new_file_name, dir)
        print(f'Copied {src_file_key} to {new_file_name} in {dir}')

    # delete a file on S3
    def del_file(self, file_name, dir=''):
        if not self.exist_bucket():
            print('Not exitst bucket')
            assert False
        if not self.exist_dir(dir):
            print('Not exitst dir')
            assert False
        self.s3_clinet.delete_object(Bucket=self.bucket_name, Key=dir+file_name)
        assert not self.check_uploaded_file(file_name, dir)
        print(f'Deleted {file_name} from {dir}')

# Test
def test_s3_utils():
    from datetime import datetime
    current_datetime = datetime.now().strftime('%Y%m%d%H%M%S')
    # Test for create bucket
    bucket_name = 'test-bucket-by-matsukage-' + current_datetime
    print(bucket_name)
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
    
if __name__ == "__main__":
    # Test
    test_s3_utils()

    



