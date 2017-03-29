import json
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


if __name__ == "__main__":
    get_elite_users_id('../data/input/yelp/yelp_academic_dataset_user.json', '../data/output/elite_users_id.txt')
