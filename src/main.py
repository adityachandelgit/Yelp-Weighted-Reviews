import json
import re
from collections import defaultdict
from time import time


def get_elite_users_id(input_path, output_path):
    t = time()
    print 'Getting elite users ID and saving them to txt...'
    with open(input_path) as infile, open(output_path, "wb") as outfile:
        for line in infile:
            json_line = json.loads(line)
            if json_line['elite'][0] != 'None':
                outfile.write(json_line['user_id'] + '\n')
    outfile.close()
    print("done in %0.3fs." % (time() - t))


def extract_review_metadata_per_user(input_path, output_path):
    user_review = defaultdict(list)
    with open(input_path) as infile, open(output_path, "wb") as outfile:
        for line in infile:
            review = json.loads(line)
            metadata = {}
            metadata["review_id"] = review["review_id"]
            metadata["business_id"] = review["business_id"]
            metadata["stars"] = review["stars"]
            metadata["review_len"] = len(re.findall(r'\w+', review["text"]))
            user_review[review["user_id"]].append(metadata)
        json.dump(user_review, outfile)
    outfile.close()


def merge_review_metadata_with_user(input_path, output_path, users_path):
    with open(input_path) as infile:
        review_metadata_per_user = json.load(infile)

    with open(users_path) as user_file, open(output_path, "wb") as outfile:
        for line in user_file:
            user = json.loads(line)
            user_id = user["user_id"]
            if review_metadata_per_user.has_key(user_id):
                user["review_metadata"] = review_metadata_per_user[user_id]
            else:
                user["review_metadata"] = ""
            json.dump(user, outfile)
        outfile.close()


if __name__ == "__main__":
    get_elite_users_id('../data/input/yelp/yelp_academic_dataset_user.json', '../data/output/elite_users_id.txt')
    extract_review_metadata_per_user('../data/input/review_try.json', '../data/temp/review_metadata_per_user.json')
    merge_review_metadata_with_user('../data/temp/review_metadata_per_user.json', '../data/output/user_with_review.json', '../data/input/user_try.json')