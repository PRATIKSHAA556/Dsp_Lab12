import base64

# Encode function logic in base64
code = """
def add(a, b):
    return a + b
"""

# Obfuscate (encode)
encoded = base64.b64encode(code.encode()).decode()

# Decode and execute later
exec(base64.b64decode(encoded).decode())

print(add(5, 10))  # Output: 15
