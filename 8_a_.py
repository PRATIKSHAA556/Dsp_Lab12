from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization

# 1. Generate RSA Key Pair
private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
public_key = private_key.public_key()

# 2. Message to sign
message = b"Secure transaction data"

# 3. Generate Digital Signature
signature = private_key.sign(
    message,
    padding.PSS(
        mgf=padding.MGF1(hashes.SHA256()), 
        salt_length=padding.PSS.MAX_LENGTH
    ),
    hashes.SHA256()
)

print("Generated Signature:", signature.hex())

# 4. Verify Digital Signature
try:
    public_key.verify(
        signature,
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()), 
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    print("✅ Signature is valid.")
except Exception:
    print("❌ Signature is invalid.")
