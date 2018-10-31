""" Simple job that reads from a storage bucket """

import os
import re
import json
from google.cloud import storage
from google.oauth2 import service_account

SERVICE_ACCOUNT_KEY_LOCATION = os.environ['SERVICE_ACCOUNT_KEY_LOCATION']
BUCKET_NAME = os.environ['BUCKET_NAME']

MODEL_DIR = 'models'
EVALUATION_DIR = 'evaluations'


def run():
    """ Get the models from GCS and then have them play eachother. """
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_KEY_LOCATION)
    scoped_credentials = credentials.with_scopes(
        ['https://www.googleapis.com/auth/cloud-platform'])

    # Use the hand-crafted GCS client
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.get_bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=MODEL_DIR)

    models = []
    seen_models = set()
    model_reg = re.compile('\d{6}-\w+')
    for b in blobs:
        match = model_reg.search(b.name)
        if match and not match.group(0) in seen_models:
            seen_models.add(match.group(0))
            models.append(match.group(0))

    # Now that we have all the models, we can pit them against eachother them.
    # For now, just pick the last two.
    p1, p2 = None, None
    if len(models) == 0:
        sys.stderr.write('No models found!\n')
        sys.exit(1)
    elif len(models) == 1:
        p1, p2 = models[0], models[0]
    else:
        p1, p2 = models[-1], models[-2]

    play_matches(p1, p2)


def play_matches(m1, m2)
    """ Play matches against two models """
    pass


def print_env():
    flags = {
        'BUCKET_NAME': BUCKET_NAME,
        'SERVICE_ACCOUNT_KEY_LOCATION': SERVICE_ACCOUNT_KEY_LOCATION,
    }
    print("Env variables are:")
    print('\n'.join('--{}={}'.format(flag, value)
                    for flag, value in flags.items()))


if __name__ == '__main__':
    print_env()
    run()
