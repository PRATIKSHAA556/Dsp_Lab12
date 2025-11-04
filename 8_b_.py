# from flask import Flask, request, jsonify
# import jwt
# import datetime

# app = Flask(__name__)

# # Secret key for JWT encoding/decoding
# app.config["SECRET_KEY"] = "supersecretkey"

# # Simulated user database
# users = {"admin": "password123"}

# # -------------------------
# # Login Route (Authentication)
# # -------------------------
# @app.route("/login", methods=["POST"])
# def login():
#     data = request.get_json()  # <-- safer than request.json
#     if not data or "username" not in data or "password" not in data:
#         return jsonify({"error": "Missing username or password"}), 400

#     username = data["username"]
#     password = data["password"]

#     if username in users and users[username] == password:
#         # Create JWT token
#         token = jwt.encode(
#             {
#                 "user": username,
#                 "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=30)
#             },
#             app.config["SECRET_KEY"],
#             algorithm="HS256"
#         )

#         # jwt.encode returns bytes in some versions → convert to str
#         if isinstance(token, bytes):
#             token = token.decode("utf-8")

#         return jsonify({"token": token})

#     return jsonify({"error": "Invalid credentials"}), 401


# # -------------------------
# # Secure API (Authorization with JWT)
# # -------------------------
# @app.route("/secure-data", methods=["GET"])
# def secure_data():
#     token = request.headers.get("Authorization")
#     if not token:
#         return jsonify({"error": "Token missing"}), 403

#     try:
#         decoded = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
#         return jsonify({"message": f"Hello {decoded['user']}, here is your secure data!"})
#     except jwt.ExpiredSignatureError:
#         return jsonify({"error": "Token expired"}), 401
#     except jwt.InvalidTokenError:
#         return jsonify({"error": "Invalid token"}), 401


# # -------------------------
# # Main
# # -------------------------
# if __name__ == "__main__":
#     app.run(debug=True)































from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
import jwt
import datetime


app = FastAPI()


SECRET_KEY = "supersecretkey"
ALGORITHM = "HS256"


users = {"admin": "password123"}



class LoginRequest(BaseModel):
    username: str
    password: str



@app.post("/login")
def login(request: LoginRequest):
    username = request.username
    password = request.password

    if username in users and users[username] == password:
    
        token = jwt.encode(
            {
                "user": username,
                "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=30)
            },
            SECRET_KEY,
            algorithm=ALGORITHM
        )

        
        if isinstance(token, bytes):
            token = token.decode("utf-8")

        return {"token": token}

    raise HTTPException(status_code=401, detail="Invalid credentials")



def verify_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=403, detail="Token missing")

    try:
        decoded = jwt.decode(authorization, SECRET_KEY, algorithms=[ALGORITHM])
        return decoded
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")



@app.get("/secure-data")
def secure_data(decoded: dict = Depends(verify_token)):
    return {"message": f"Hello {decoded['user']}, here is your secure data!"}

