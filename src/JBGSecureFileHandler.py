import base64
import io
import secrets
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from src.JBGLogger import JBGLogger

class SecureFileHandler:
    def __init__(self, encryption_key: str):
        self.logger = JBGLogger(level="INFO").logger
        self.key = base64.b64decode(encryption_key)  # Expect 256-bit key (32 bytes)
        self.nonce_size = 12  # AES-GCM standard

    def encrypt_file(self, input_path: str, encrypted_path: str):
        """Krypterar en fil med AES-GCM och sparar till disk"""
        with open(input_path, "rb") as f:
            plaintext = f.read()
        aesgcm = AESGCM(self.key)
        nonce = secrets.token_bytes(self.nonce_size)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)
        with open(encrypted_path, "wb") as f:
            f.write(nonce + ciphertext)
        self.logger.info(f"Fil krypterad och sparad som {encrypted_path} ({len(ciphertext)} bytes)")

    def decrypt_file_to_memory(self, encrypted_path: str) -> io.BytesIO:
        """Dekrypterar AES-GCM-fil till BytesIO-ström"""
        with open(encrypted_path, "rb") as f:
            data = f.read()
        nonce = data[:self.nonce_size]
        ciphertext = data[self.nonce_size:]
        aesgcm = AESGCM(self.key)
        try:
            decrypted = aesgcm.decrypt(nonce, ciphertext, None)
        except Exception as e:
            self.logger.error(f"AES-GCM avkryptering misslyckades: {e}")
            raise
        decrypted_data = io.BytesIO(decrypted)
        decrypted_data.seek(0)
        self.logger.info(f"Avkrypterad stream har längd {len(decrypted)} bytes")
        return decrypted_data

    def encrypt_text(self, text: str, encrypted_path: str):
        """Krypterar textinnehåll och sparar till fil"""
        aesgcm = AESGCM(self.key)
        nonce = secrets.token_bytes(self.nonce_size)
        ciphertext = aesgcm.encrypt(nonce, text.encode("utf-8"), None)
        with open(encrypted_path, "wb") as f:
            f.write(nonce + ciphertext)
        self.logger.info(f"Text krypterad och sparad som {encrypted_path}")

    def decrypt_text_from_file(self, encrypted_path: str) -> str:
        """Läser och dekrypterar krypterad textfil"""
        with open(encrypted_path, "rb") as f:
            data = f.read()
        nonce = data[:self.nonce_size]
        ciphertext = data[self.nonce_size:]
        aesgcm = AESGCM(self.key)
        try:
            plaintext = aesgcm.decrypt(nonce, ciphertext, None).decode("utf-8")
        except Exception as e:
            self.logger.error(f"Misslyckad dekryptering av textfil: {e}")
            raise
        return plaintext

    def decrypt_bytesio(self, encrypted_path: str) -> io.BytesIO:
        """Wrapper för att tydliggöra att vi jobbar mot ljud"""
        return self.decrypt_file_to_memory(encrypted_path)

    def encrypt_bytesio(self, input_stream: io.BytesIO, encrypted_path: str):
        """Kryptera en BytesIO-ström och skriv till fil"""
        plaintext = input_stream.getvalue()
        aesgcm = AESGCM(self.key)
        nonce = secrets.token_bytes(self.nonce_size)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)
        with open(encrypted_path, "wb") as f:
            f.write(nonce + ciphertext)
        self.logger.info(f"Stream krypterad och sparad som {encrypted_path}")