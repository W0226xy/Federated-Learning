import random

from Crypto import Random
from tqdm import tqdm
import base64
from Crypto.PublicKey import RSA
from Crypto.Signature import PKCS1_v1_5 as PKCS1_signature
from Crypto.Cipher import PKCS1_v1_5 as PKCS1_cipher
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Hash import SHA256


def generate_key():  # Generate a pair of RSA keys
    random_generator = Random.new().read
    rsa = RSA.generate(2048, random_generator)
    public_key = rsa.publickey().exportKey()  # Export public key in PEM format
    private_key = rsa.exportKey()  # Export private key in PEM format

    with open('rsa_private_key.pem', 'wb') as f:
        f.write(private_key)

    with open('rsa_public_key.pem', 'wb') as f:
        f.write(public_key)


def get_key(key_file):  # Read RSA public or private key from a file
    with open(key_file) as f:
        data = f.read()
        key = RSA.importKey(data)
    return key


def sign(msg):  # Sign a message
    private_key = get_key('rsa_private_key.pem')
    signer = PKCS1_signature.new(private_key)
    digest = SHA256.new()
    digest.update(bytes(msg.encode("utf8")))
    return signer.sign(digest)


def verify(msg, signature):  # Verify a signature
    pub_key = get_key('rsa_public_key.pem')
    signer = PKCS1_signature.new(pub_key)
    digest = SHA256.new()
    digest.update(bytes(msg.encode("utf8")))
    return signer.verify(digest, signature)


def perturb_items(item_ids, perturbation_rate=0.2):
    """
    Perturb the list of items by randomly adding and removing some items.

    :param item_ids: Original list of items
    :param perturbation_rate: Rate of perturbation, default is 20%
    :return: Perturbed list of items
    """
    perturbed_items = set(item_ids)

    # Randomly remove a certain proportion of items
    num_items_to_remove = int(len(item_ids) * perturbation_rate)
    items_to_remove = random.sample(perturbed_items, min(num_items_to_remove, len(perturbed_items)))
    perturbed_items.difference_update(items_to_remove)

    # Randomly add some fake item IDs
    num_items_to_add = int(len(item_ids) * perturbation_rate)
    fake_items = {f"fake_item_{random.randint(1000, 9999)}" for _ in range(num_items_to_add)}
    perturbed_items.update(fake_items)

    return list(perturbed_items)


def encrypt_data(msg):  # Encrypt a message using the public key
    pub_key = get_key('rsa_public_key.pem')
    cipher = PKCS1_OAEP.new(pub_key)
    encrypt_text = base64.b64encode(cipher.encrypt(bytes(msg.encode("utf8"))))
    return encrypt_text.decode('utf-8')


def decrypt_data(encrypt_msg):  # Decrypt a message using the private key
    private_key = get_key('rsa_private_key.pem')
    cipher = PKCS1_OAEP.new(private_key)
    back_text = cipher.decrypt(base64.b64decode(encrypt_msg))
    return back_text.decode('utf-8')


def encrypt_user_data(user_data):
    """
    Encrypt user data (e.g., embeddings) using the public key.

    :param user_data: User data to encrypt
    :return: Encrypted user data
    """
    encrypted_data = []
    pub_key = get_key('rsa_public_key.pem')
    cipher = PKCS1_OAEP.new(pub_key)
    for data in user_data:
        encrypted_data.append(base64.b64encode(cipher.encrypt(bytes(str(data).encode("utf8")))).decode('utf-8'))
    return encrypted_data


def decrypt_user_data(encrypted_data):
    """
    Decrypt user data (e.g., embeddings) using the private key.

    :param encrypted_data: Encrypted user data
    :return: Decrypted user data
    """
    decrypted_data = []
    private_key = get_key('rsa_private_key.pem')
    cipher = PKCS1_OAEP.new(private_key)
    for data in encrypted_data:
        decrypted_data.append(cipher.decrypt(base64.b64decode(data)).decode('utf-8'))
    return decrypted_data

# 1. Key Generation: Generate RSA keys using generate_key() and save them in files.
# 2. Sign and Verify: Use private key to sign messages with sign() and verify the signature using the public key with verify().
# 3. Encrypt and Decrypt: Encrypt data with encrypt_data() using the public key and decrypt with decrypt_data() using the private key.
# 4. Encrypt and Decrypt User Data: Encrypt user embeddings using encrypt_user_data() and decrypt them using decrypt_user_data().