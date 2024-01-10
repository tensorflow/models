import boto3

s3 = boto3.resource('s3')
bucket_names = ['input-image-for-detection', 'output-from-detection']

for bucket_name in bucket_names:
    bucket = s3.Bucket(bucket_name)
    bucket.create()

    name_list = [b.name for b in s3.buckets.all()]
    if bucket_name in name_list:
        print(f'Exist {bucket_name}')
        bucket.delete()
    else:
        print(f'Faled to create {bucket_name}')

