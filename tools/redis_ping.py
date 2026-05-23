from redis import StrictRedis


def main():
    client = StrictRedis(host="127.0.0.1", port=6379, db=0)
    print(client.ping())


if __name__ == "__main__":
    main()
