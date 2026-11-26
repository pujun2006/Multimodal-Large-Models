import base64
import hashlib
import hmac
import json
import os
import time
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


# 1. 密钥派生与对称加解密（使用 12 字节标准 GCM Nonce，并分离盐值与 Nonce）
def derive_aes_key(secret: bytes, salt: bytes, iterations: int = 100_000) -> bytes:
    """当输入为口令或低熵秘密时，通过 PBKDF2-HMAC-SHA256 派生 32 字节 AES-256 密钥。"""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
    )
    return kdf.derive(secret)


def encrypt_payload_aesgcm(plaintext: bytes, aes_key: bytes, aad: bytes = b"") -> tuple[bytes, bytes]:
    """使用 AES-256-GCM 加密数据；返回 (nonce_12B, ciphertext_with_tag)。"""
    if len(aes_key) != 32:
        raise ValueError("AES-256 requires a 32-byte key.")
    nonce = os.urandom(12)  # AES-GCM 标准推荐 12 字节随机 Nonce
    aesgcm = AESGCM(aes_key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, aad)
    return nonce, ciphertext


def decrypt_payload_aesgcm(nonce: bytes, ciphertext: bytes, aes_key: bytes, aad: bytes = b"") -> bytes:
    """使用 AES-256-GCM 解密并校验完整性（含 AAD）。"""
    aesgcm = AESGCM(aes_key)
    return aesgcm.decrypt(nonce, ciphertext, aad)


# 2. RSA-OAEP 密钥封装与数字签名（用于安全分发对称密钥与校验分发来源/防篡改）
def generate_rsa_keypair(key_size: int = 2048) -> tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
    return private_key, private_key.public_key()


def wrap_key_rsa_oaep(recipient_public_key: rsa.RSAPublicKey, raw_key: bytes) -> bytes:
    return recipient_public_key.encrypt(
        raw_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )


def unwrap_key_rsa_oaep(recipient_private_key: rsa.RSAPrivateKey, wrapped_key: bytes) -> bytes:
    return recipient_private_key.decrypt(
        wrapped_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )


def sign_package(sender_private_key: rsa.RSAPrivateKey, message: bytes) -> bytes:
    return sender_private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )


def verify_signature(sender_public_key: rsa.RSAPublicKey, message: bytes, signature: bytes) -> bool:
    try:
        sender_public_key.verify(
            signature,
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return True
    except Exception:
        return False


# 3. 安全分发数据包打包与校验解包（含有效期、接收方标识与防篡改校验）
def create_distribution_package(
    plaintext: bytes,
    recipient_id: str,
    recipient_public_key: rsa.RSAPublicKey,
    sender_private_key: rsa.RSAPrivateKey,
    ttl_seconds: int = 3600,
    passphrase_or_seed: bytes | None = None,
) -> dict:
    # 若提供口令/种子则派生；否则直接生成 32 字节高熵会话 DEK，避免对随机密钥做冗余 PBKDF2
    if passphrase_or_seed is not None:
        salt = os.urandom(16)
        dek = derive_aes_key(passphrase_or_seed, salt)
        kdf_meta = {"mode": "pbkdf2-sha256", "salt": base64.b64encode(salt).decode("ascii")}
    else:
        dek = os.urandom(32)
        kdf_meta = {"mode": "direct-random-256", "salt": None}

    issued_at = int(time.time())
    expires_at = issued_at + ttl_seconds
    header = {
        "version": 1,
        "recipient_id": recipient_id,
        "issued_at": issued_at,
        "expires_at": expires_at,
        "kdf": kdf_meta,
    }
    aad_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")

    nonce, ciphertext = encrypt_payload_aesgcm(plaintext, dek, aad=aad_bytes)
    wrapped_dek = wrap_key_rsa_oaep(recipient_public_key, dek)

    package_body = {
        "header": header,
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        "wrapped_dek": base64.b64encode(wrapped_dek).decode("ascii"),
    }
    canonical_body = json.dumps(package_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature = sign_package(sender_private_key, canonical_body)

    return {
        "package": package_body,
        "signature": base64.b64encode(signature).decode("ascii"),
    }


def receive_and_decrypt_package(
    envelope: dict,
    expected_recipient_id: str,
    recipient_private_key: rsa.RSAPrivateKey,
    sender_public_key: rsa.RSAPublicKey,
    max_clock_skew: int = 10,
) -> bytes:
    package_body = envelope["package"]
    signature = base64.b64decode(envelope["signature"])
    canonical_body = json.dumps(package_body, sort_keys=True, separators=(",", ":")).encode("utf-8")

    # 1) 验证发送方签名，确认来源可信且包体未被篡改
    if not verify_signature(sender_public_key, canonical_body, signature):
        raise ValueError("Signature verification failed: package may be forged or tampered.")

    header = package_body["header"]
    # 2) 校验接收方身份与分发时效（防过期/重放）
    if header.get("recipient_id") != expected_recipient_id:
        raise PermissionError("Recipient ID mismatch.")

    now = int(time.time())
    if now > header.get("expires_at", 0) + max_clock_skew:
        raise ValueError("Distribution package has expired.")

    # 3) 解封装对称密钥并解密载荷（AAD 参与 GCM 认证）
    wrapped_dek = base64.b64decode(package_body["wrapped_dek"])
    dek = unwrap_key_rsa_oaep(recipient_private_key, wrapped_dek)

    nonce = base64.b64decode(package_body["nonce"])
    ciphertext = base64.b64decode(package_body["ciphertext"])
    aad_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")

    return decrypt_payload_aesgcm(nonce, ciphertext, dek, aad=aad_bytes)


def main():
    # 模拟发送方与接收方各自持有密钥对（生产环境应从安全存储/KMS 加载 PEM）
    sender_priv, sender_pub = generate_rsa_keypair()
    recipient_priv, recipient_pub = generate_rsa_keypair()

    sensitive_multimodal_payload = b"Multimodal embedding & sensitive metadata package."
    recipient_id = "node-edge-hk-01"

    # 发送方打包与签名
    envelope = create_distribution_package(
        plaintext=sensitive_multimodal_payload,
        recipient_id=recipient_id,
        recipient_public_key=recipient_pub,
        sender_private_key=sender_priv,
        ttl_seconds=600,
    )

    # 模拟序列化后传输（例如通过 HTTPS / 消息队列发送 JSON 字符串）
    wire_bytes = json.dumps(envelope).encode("utf-8")
    payload_sha256 = hashlib.sha256(wire_bytes).hexdigest()
    print(f"Wire size: {len(wire_bytes)} bytes | SHA-256: {payload_sha256[:16]}...")

    # 接收方验签、校验策略并解密
    received_envelope = json.loads(wire_bytes.decode("utf-8"))
    restored_data = receive_and_decrypt_package(
        envelope=received_envelope,
        expected_recipient_id=recipient_id,
        recipient_private_key=recipient_priv,
        sender_public_key=sender_pub,
    )

    assert hmac.compare_digest(
        hashlib.sha256(restored_data).digest(),
        hashlib.sha256(sensitive_multimodal_payload).digest(),
    )
    print("Decrypted successfully:", restored_data.decode("utf-8"))


if __name__ == "__main__":
    main()