import boto3

class s3_utils:
    s3_resource = boto3.resource('s3')
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
        bucket.create()
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

# Test
def test_s3_utils():
    # Test for create bucket
    bucket_name = 'test-bucket-by-matsukage'
    s3 = s3_utils(bucket_name)
    assert s3.exist_bucket()
    
    # Test for create directory
    dir_name = 'test-dir-by-matsukage/'
    s3.mk_dir(dir_name)
    assert s3.exist_dir(dir_name)

    # Test for delete directory
    s3.del_dir(dir_name)
    assert not s3.exist_dir(dir_name)

    # Test for delete buckets
    s3.del_bucket()
    assert not s3.exist_bucket()

    print('Success')
    
if __name__ == "__main__":
    test_s3_utils()

    



