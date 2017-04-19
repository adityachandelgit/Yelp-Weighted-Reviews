import json
import re
from collections import defaultdict
from time import time
import numpy as np


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
            outfile.write('\n')
        outfile.close()


def calculate_rating_variance(input_path, output_path):

    with open(input_path) as infile, open(output_path, "wb") as outfile:
        for line in infile:
            user = json.loads(line)
            ratings = []
            for review in user["review_metadata"]:
                print(int(review["stars"]))
                ratings.append(int(review["stars"]))
            user["rating_variance"] = np.var(ratings)
            json.dump(user, outfile)
            outfile.write('\n')
    outfile.close()


def businesses_as_dict(input_path, output_path):
    businesses = {}
    with open(input_path) as infile, open(output_path, "wb") as outfile:
        for line in infile:
            business = json.loads(line)
            print business
            businesses[business["business_id"]] = business
        json.dump(businesses, outfile)
    outfile.close()


def get_review_category(business_path, input_path, output_path):
    with open(business_path) as infile:
        businesses = json.load(infile)

    with open(input_path) as user_file, open(output_path, "wb") as outfile:
        for line in user_file:
            user = json.loads(line)
            reviews_in_category = {}
            for review in user["review_metadata"]:
                business_id = review["business_id"]

                if businesses.has_key(business_id):
                    business = businesses[business_id]
                    categories = business["categories"]

    #               add 1 to each category and store in a dict
                    for business_type in categories:
                        if reviews_in_category.has_key(business_type):
                            reviews_in_category[business_type] += 1
                        else:
                            reviews_in_category[business_type] = 1
            user["reviews_in_category"] = reviews_in_category
            json.dump(user, outfile)
            outfile.write('\n')
    outfile.close()


if __name__ == "__main__":
    # get_elite_users_id('../data/input/yelp/yelp_academic_dataset_user.json', '../data/output/elite_users_id.txt')
    # extract_review_metadata_per_user('../data/input/review_try.json', '../data/temp/review_metadata_per_user.json')
    # merge_review_metadata_with_user('../data/temp/review_metadata_per_user.json', '../data/temp/user_with_review.json', '../data/input/user_try.json')
    # calculate_rating_variance('../data/temp/user_with_review.json', '../data/temp/user_with_review_var.json')
    # businesses_as_dict('../data/input/business_try.json', '../data/temp/yelp_business_as_dict.json')
    get_review_category('../data/temp/yelp_business_as_dict.json', '../data/temp/user_with_review_var.json', '../data/output/user_with_category_reviews.json')