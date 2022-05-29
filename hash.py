import argparse
from hashlib import sha1

def compute_hash(email):
    return sha1(email.lower().encode('utf-8')).hexdigest()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--email",
        default="email@domain",
        help="the email to be hashed"
    )
    args = parser.parse_args()

    print(compute_hash(args.email))