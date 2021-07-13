python generate_tfrecord.py --csv_input=CAPTCHA_images\train_labels.csv --image_dir=CAPTCHA_images\train --output_path=CAPTCHA_images\train.record
python generate_tfrecord.py --csv_input=CAPTCHA_images\test_labels.csv --image_dir=CAPTCHA_images\test --output_path=CAPTCHA_images\test.record
pause