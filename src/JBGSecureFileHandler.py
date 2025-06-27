import pyAesCrypt
import io
import os

class SecureFileHandler:
    def __init__(self, encryption_key: str, buffer_size: int = 64 * 1024):
        if not encryption_key:
            raise ValueError("Encryption key must not be empty.")
        self.key = encryption_key
        self.buffer_size = buffer_size

    def encrypt_file(self, input_path: str, encrypted_path: str):
        """Krypterar en fil på disk"""
        pyAesCrypt.encryptFile(input_path, encrypted_path, self.key, self.buffer_size)

    def decrypt_file_to_memory(self, encrypted_path: str) -> io.BytesIO:
        """Dekrypterar en fil från disk till ett minnesobjekt (BytesIO)"""
        file_size = os.path.getsize(encrypted_path)
        decrypted_data = io.BytesIO()
        with open(encrypted_path, "rb") as f_in:
            pyAesCrypt.decryptStream(f_in, decrypted_data, self.key, self.buffer_size, file_size)
        decrypted_data.seek(0)
        return decrypted_data

    def encrypt_text(self, plain_text: str, encrypted_path: str):
        """Krypterar en textsträng och sparar den krypterat"""
        tmp_path = encrypted_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as tmp:
            tmp.write(plain_text)
        self.encrypt_file(tmp_path, encrypted_path)
        os.remove(tmp_path)

    def decrypt_text_from_file(self, encrypted_path: str) -> str:
        """Dekrypterar en textfil från disk och returnerar som sträng"""
        decrypted = self.decrypt_file_to_memory(encrypted_path)
        return decrypted.read().decode("utf-8")

    def encrypt_bytesio(self, data: io.BytesIO, encrypted_path: str):
        """Kryptera en BytesIO-ström (användbar för mp3-data efter upload)"""
        with open(encrypted_path, "wb") as f_out:
            data.seek(0)
            pyAesCrypt.encryptStream(data, f_out, self.key, self.buffer_size)

    def decrypt_bytesio(self, encrypted_path: str) -> io.BytesIO:
        """Dekryptera till BytesIO (t.ex. för transkribering)"""
        return self.decrypt_file_to_memory(encrypted_path)
