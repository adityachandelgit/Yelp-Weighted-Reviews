import csv
import json
import random
import re
from collections import defaultdict
from time import time
import numpy as np
from datetime import date

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
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

                    # Add 1 to each category and store in a dict
                    if categories is not None:
                        for business_type in categories:
                            if reviews_in_category.has_key(business_type):
                                reviews_in_category[business_type] += 1
                            else:
                                reviews_in_category[business_type] = 1
            user["reviews_in_category"] = reviews_in_category
            json.dump(user, outfile)
            outfile.write('\n')
    outfile.close()


def extract_elite_users_features(input_path, path_elite_users_features, review_category):
    elite_users_features = []
    with open(input_path) as user_file:
        for line in user_file:
            user = json.loads(line)

            an_elite_feature = [len(user["friends"]), user["fans"], user['review_count'], user['rating_variance'],
                                user["avg_review_len"], user["average_stars"]]

            # an_elite_feature = [len(user["friends"]), user['review_count']]

            total_review_len = 0
            count_reviews = 0
            for review in user['review_metadata']:
                count_reviews += 1
                total_review_len += review['review_len']
            an_elite_feature.append(float(total_review_len) / float(count_reviews))

            date_split = user['yelping_since'].split('-')
            d0 = date(int(date_split[0]), int(date_split[1]), int(date_split[2]))
            d1 = date.today()
            delta = d1 - d0
            an_elite_feature.append(delta.days)

            percentage = 0.0
            if user['reviews_in_category'].has_key(review_category):
                wanted_review_cat_count = user['reviews_in_category'][review_category]
                total_review_cat_count = 0
                for review_cat in user['reviews_in_category']:
                    total_review_cat_count += user['reviews_in_category'][review_cat]
                percentage = 100 * float(wanted_review_cat_count) / float(total_review_cat_count)
                an_elite_feature.append(int(percentage))
            else:
                an_elite_feature.append(0)

            if user['elite'][0] != 'None' and user['reviews_in_category'].has_key(review_category) and percentage > 25:
                # elite_users_features.append({user['user_id']: an_elite_feature})
                elite_users_features.append(
                    json.dumps({'elite': 1, 'user_id': user['user_id'], 'features': an_elite_feature}))
            else:
                elite_users_features.append(
                    json.dumps({'elite': 0, 'user_id': user['user_id'], 'features': an_elite_feature}))

    with open(path_elite_users_features, "wb") as f1:
        for feature in elite_users_features:
            f1.write("%s\n" % feature)


def classifier(user_features):
    features_elite = []
    features_non_elite = []
    with open(user_features, 'rb') as f:
        for line in f:
            is_elite = json.loads(line)['elite']
            if is_elite == 1:
                features_elite.append(json.loads(line)['features'])
            elif is_elite == 0:
                features_non_elite.append(json.loads(line)['features'])

    random.shuffle(features_non_elite)

    data1 = np.array(features_elite)
    labels1 = np.array([1] * len(features_elite))

    data2 = np.array(features_non_elite[0: len(features_non_elite) - 1])
    labels2 = np.array([0] * len(data2))

    data = np.concatenate((data1, data2), axis=0)
    labels = np.concatenate((labels1, labels2), axis=0)

    names = ["Nearest Neighbors",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]
    classifiers = [
        KNeighborsClassifier(3),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()
    ]

    with open('Rest_Elites.txt', 'wb') as rest_elites:
        for idx, val in enumerate(classifiers):
            clf = val.fit(data, labels)

            count1 = 0
            count0 = 0
            for feature_non_elite in features_non_elite:
                if clf.predict(feature_non_elite)[0].item() == 1:
                    count1 += 1
                    print count1
                    rest_elites.write(str(count1) + '\n')
                else:
                    count0 += 1
            print count1
            print count0

            print names[idx] + ': ' + str(clf.score(data, labels))
        print 'Finished running the classification algorithms.\n'


def extract_features_with_label(input_path, output_path):
    with open(input_path) as infile, open(output_path, "wb") as outfile:
        for line in infile:
            user = json.loads(line)
            features = user["features"]
            features.append(user["elite"])
            record = np.asarray(features)
            record.tofile(outfile, sep=',')
            outfile.write("\n")
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


def reviews_per_business(input_path, output_path):
    review_of_a_business = defaultdict(list)
    with open(input_path) as infile, open(output_path, "wb") as outfile:
        for line in infile:
            review = json.loads(line)
            metadata = {}
            metadata["review_id"] = review["review_id"]
            metadata["user_id"] = review["business_id"]
            metadata["stars"] = review["stars"]
            review_of_a_business[review["business_id"]].append(metadata)
        json.dump(review_of_a_business, outfile)
    outfile.close()


def extract_features_with_label(input_path, output_path):
    with open(input_path) as infile, open(output_path, "wb") as outfile:
        for line in infile:
            user = json.loads(line)
            features = user["features"]
            features.append(user["elite"])
            record = np.asarray(features)
            record.tofile(outfile, sep=',')
            outfile.write("\n")
        outfile.close()


if __name__ == "__main__":
    # get_elite_users_id('../data/input/yelp/yelp_academic_dataset_user.json', '../data/output/elite_users_id.txt')
    # extract_review_metadata_per_user('../data/input/review_try.json', '../data/temp/review_metadata_per_user.json')
    # merge_review_metadata_with_user('../data/temp/review_metadata_per_user.json', '../data/temp/user_with_review.json', '../data/input/user_try.json')
    # calculate_rating_variance_review_len('../data/temp/user_with_review.json', '../data/temp/user_with_review_var.json')
    # businesses_as_dict('../data/input/business_try.json', '../data/temp/yelp_business_as_dict.json')
    # get_review_category('../data/temp/yelp_business_as_dict.json', '../data/temp/user_with_review_var.json',
    #                     '../data/output/user_with_category_reviews.json')
    # extract_elite_users_features('../data/output/user_with_category_reviews.json', '../data/output/users_features.csv', 'Restaurants')

    classifier('../data/output/users_features.csv')
    # extract_features_with_label('../data/output/users_features.csv', '../data/output/feature_with_labels.csv')


    # get_review_category('../data/temp/yelp_business_as_dict.json', '../data/temp/user_with_review_var.json', '../data/output/user_with_category_reviews.json')
    # get_user_training_dataframe('../data/output/user_with_category_reviews.json', '../data/output/training.csv',
    #                             "Restaurants")
    # reviews_per_business('../data/input/review_try.json', '../data/temp/review_metadata_per_business.json')
    extract_features_with_label('../data/output/usersfeatry.csv', '../data/output/feature_with_labels.csv')
