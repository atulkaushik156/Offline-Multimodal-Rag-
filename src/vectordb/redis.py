import redis



redis_client = redis.Redis(
    host='localhost', # Change this to your Redis server IP or URL in production
    port=6379,        # Default Redis port
    db=0,             # Default database index (0-15 available)
    decode_responses=True # Industry standard: returns strings instead of raw bytes
)