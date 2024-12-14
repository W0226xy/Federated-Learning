import random
import numpy as np
from Crypto import Random
from tqdm import tqdm
import base64
from Crypto.PublicKey import RSA
from Crypto.Signature import PKCS1_v1_5 as PKCS1_signature
from Crypto.Cipher import PKCS1_v1_5 as PKCS1_cipher
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Hash import SHA256

def generate_key():  # 生成一堆RSA密钥
    random_generator = Random.new().read  # 使用 RSA.generate() 生成一个2048位的RSA密钥对。
    rsa = RSA.generate(2048, random_generator)
    public_key = rsa.publickey().exportKey()  # 将公钥导出成PEM格式并保存到 rsa_public_key.pem 文件
    private_key = rsa.exportKey()  # rsa.exportKey() 将私钥导出成PEM格式并保存到 rsa_private_key.pem 文件中。

    with open('rsa_private_key.pem', 'wb') as f:
        f.write(private_key)

    with open('rsa_public_key.pem', 'wb') as f:
        f.write(public_key)

def get_key(key_file):  # 从指定文件中读取RSA公钥或私钥。
    with open(key_file) as f:  # 打开给定的密钥文件并读取内容。
        data = f.read()
        key = RSA.importKey(data)  # 使用 RSA.importKey() 导入密钥，返回一个RSA密钥对象，之后可以在加密、解密、签名和验证操作中使用。
    return key

def sign(msg):  # 对消息进行签名。消息签名可以用来证明消息的发送者，并保证消息未被篡改。
    private_key = get_key('rsa_private_key.pem')  # 读取私钥 rsa_private_key.pem。
    signer = PKCS1_signature.new(private_key)  # 创建 PKCS1_signature 对象，使用SHA-256哈希函数对消息进行哈希处理
    digest = SHA256.new()
    digest.update(bytes(msg.encode("utf8")))  # 使用私钥对哈希值进行签名并返回签名结果。
    return signer.sign(digest)

def verify(msg, signature):  # 验证签名。
    # use signature because the rsa encryption lib adds salt defaultly
    pub_key = get_key('rsa_public_key.pem')  # 读取公钥 rsa_public_key.pem。
    signer = PKCS1_signature.new(pub_key)  # 使用SHA-256对消息进行哈希处理，并通过公钥对签名进行验证。
    digest = SHA256.new()
    digest.update(bytes(msg.encode("utf8")))  # 如果签名与消息匹配，则返回 True，否则返回 False。
    return signer.verify(digest, signature)

def perturb_items(item_ids, all_item_ids, epsilon=1.0):
    """
    使用局部差分隐私 (LDP) 对交互物品进行扰动。
    :param item_ids: 原始物品列表
    :param all_item_ids: 全局物品ID列表
    :param epsilon: 隐私预算参数，值越大隐私保护越弱。
    :return: 扰动后的物品列表
    """
    perturbed_items = []

    for item_id in item_ids:
        # 生成一个噪声概率，用于决定是否保留或更改当前物品
        keep_probability = np.exp(epsilon) / (np.exp(epsilon) + 1)
        if random.random() < keep_probability:
            # 保留原始物品
            perturbed_items.append(item_id)
        else:
            # 从全局物品ID中随机选择一个物品
            fake_item = random.choice(all_item_ids)
            perturbed_items.append(fake_item)

    return perturbed_items

def encrypt_data(msg):  # 使用公钥对消息进行加密。
    pub_key = get_key('rsa_public_key.pem')  # 读取公钥 rsa_public_key.pem。
    cipher = encryptor = PKCS1_OAEP.new(pub_key)  # 创建 PKCS1_OAEP 加密对象（该对象默认使用SHA-256作为哈希函数）。
    encrypt_text = base64.b64encode(cipher.encrypt(bytes(msg.encode("utf8"))))  # 使用公钥加密消息并通过base64编码返回加密后的字符串。
    return encrypt_text.decode('utf-8')

def decrypt_data(encrypt_msg):  # 使用私钥对加密后的消息进行解密。
    private_key = get_key('rsa_private_key.pem')  # 读取私钥 rsa_private_key.pem
    cipher = PKCS1_OAEP.new(private_key)  # 创建 PKCS1_OAEP 解密对象。
    back_text = cipher.decrypt(base64.b64decode(encrypt_msg))  # 使用私钥解密通过base64编码的加密消息，并返回解密后的明文消息。
    return back_text.decode('utf-8')

# 1.密钥生成：首先通过 generate_key() 函数生成公钥和私钥，并将它们分别存储到文件中。
# 2.签名与验证：使用私钥对消息签名 sign()，生成签名。使用公钥验证签名 verify()，确保消息的完整性和来源。
# 3.加密与解密：使用公钥加密数据 encrypt_data()，将明文转换为加密的密文。使用私钥解密数据 decrypt_data()，将密文还原为明文，保证数据的保密性。
