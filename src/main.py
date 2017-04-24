import json
import re
from collections import defaultdict
from time import time
import numpy as np
import pandas as pd

from datetime import date


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


def calculate_rating_variance_review_len(input_path, output_path):
    with open(input_path) as infile, open(output_path, "wb") as outfile:
        for line in infile:
            user = json.loads(line)
            ratings = []
            review_length = []
            for review in user["review_metadata"]:
                ratings.append(int(review["stars"]))
                review_length.append(int(review["review_len"]))
            user["rating_variance"] = np.var(ratings)
            user["avg_review_len"] = np.mean(review_length)
            json.dump(user, outfile)
            outfile.write('\n')
    outfile.close()


def businesses_as_dict(input_path, output_path):
    businesses = {}
    with open(input_path) as infile, open(output_path, "wb") as outfile:
        for line in infile:
            business = json.loads(line)
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
                    if (categories is not None):
                        for business_type in categories:
                            if reviews_in_category.has_key(business_type):
                                reviews_in_category[business_type] += 1
                            else:
                                reviews_in_category[business_type] = 1
            user["reviews_in_category"] = reviews_in_category
            json.dump(user, outfile)
            outfile.write('\n')
    outfile.close()


def get_user_training_dataframe(input_path, output_path, category):
    # train = pd.DataFrame(columns=('avg_review_len', 'rating_variance', 'review_count', 'average_stars',
    # 'number_of_fans','number_of_friends','number_of_review_category','joined_since'))
    with open(input_path) as user_file, open(output_path, "wb") as outfile:
        for line in user_file:
            user = json.loads(line)
            number_of_friends = len(user["friends"])
            number_of_fans = user["fans"]
            review_count = user["review_count"]
            number_of_review_category = 0
            review_category = user["reviews_in_category"]
            if review_category.has_key(category):
                number_of_review_category = review_category[category]
                if review_count != 0:
                    number_of_review_category = (float(number_of_review_category) / review_count) * 100

            joining = user["yelping_since"].split("-")
            join_date = date(int(joining[0]), int(joining[1]), int(joining[2]))
            date_diff = date.today() - join_date
            record = np.asarray([user["avg_review_len"], user["rating_variance"], review_count, user["average_stars"],
                                number_of_fans, number_of_friends, number_of_review_category, date_diff.days])
            record.tofile(outfile, sep=',')
            outfile.write("\n")
    outfile.close()


if __name__ == "__main__":
    # get_elite_users_id('../data/input/yelp/yelp_academic_dataset_user.json', '../data/output/elite_users_id.txt')
    # extract_review_metadata_per_user('../data/input/review_try.json', '../data/temp/review_metadata_per_user.json')
    # merge_review_metadata_with_user('../data/temp/review_metadata_per_user.json', '../data/temp/user_with_review.json', '../data/input/user_try.json')
    # calculate_rating_variance_review_len('../data/temp/user_with_review.json', '../data/temp/user_with_review_var.json')
    # businesses_as_dict('../data/input/business_try.json', '../data/temp/yelp_business_as_dict.json')
    # get_review_category('../data/temp/yelp_business_as_dict.json', '../data/temp/user_with_review_var.json', '../data/output/user_with_category_reviews.json')
    get_user_training_dataframe('../data/output/user_with_category_reviews.json', '../data/output/training.csv',
                                "Restaurants")
