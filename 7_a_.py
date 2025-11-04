import hashlib

def hash_string(data: str):
    print("Original String:", data)

    # MD5
    md5_hash = hashlib.md5(data.encode()).hexdigest()
    print("MD5:     ", md5_hash)

    # SHA-1
    sha1_hash = hashlib.sha1(data.encode()).hexdigest()
    print("SHA-1:   ", sha1_hash)

    # SHA-256
    sha256_hash = hashlib.sha256(data.encode()).hexdigest()
    print("SHA-256: ", sha256_hash)

    # SHA-512
    sha512_hash = hashlib.sha512(data.encode()).hexdigest()
    print("SHA-512: ", sha512_hash)

# Example usage
a=input("Enter a string to hash: ")
hash_string(a)
