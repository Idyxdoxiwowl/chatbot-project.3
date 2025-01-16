import sys
import os
import unittest
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import process_file, generate_embedding, collection


class TestLlamaChatbot(unittest.TestCase):
    def setUp(self):
        """Настройка перед каждым тестом"""
        if "file_data" not in st.session_state:
            st.session_state["file_data"] = []

    def tearDown(self):
        """Очистка после каждого теста"""
        if os.path.exists("mock_file.txt"):
            os.remove("mock_file.txt")

    def test_process_file_valid(self):
        """Тестирование функции process_file с корректным файлом"""
        with open("mock_file.txt", "wb") as mock_file:
            mock_file.write(b"Sample text for testing.")

        with open("mock_file.txt", "rb") as mock_file:
            process_file(mock_file)

        uploaded_files = [file["file_name"] for file in st.session_state["file_data"]]
        self.assertIn("mock_file.txt", uploaded_files, "File not found in session_state['file_data']")

        result = collection.get(ids=["mock_file.txt"])
        self.assertIsNotNone(result, "File data not added to collection")
        self.assertIn("Sample text for testing.", result["documents"], "File content missing in collection")

    def test_generate_embedding(self):
        """Тестирование функции generate_embedding"""
        sample_text = "Sample embedding test text."
        embedding = generate_embedding(sample_text)

        self.assertIsInstance(embedding, list, "Embedding is not a list")
        self.assertTrue(len(embedding) > 0, "Embedding list is empty")


if __name__ == "__main__":
    unittest.main()