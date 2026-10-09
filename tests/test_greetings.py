"""Unit tests for greeting detection and conversational fast-path."""

import unittest
from experiments.ask import is_conversational_input


class TestGreetingDetection(unittest.TestCase):
    def test_greeting_detection(self):
        # Direct greetings
        self.assertTrue(is_conversational_input("Hello"))
        self.assertTrue(is_conversational_input("hello"))
        self.assertTrue(is_conversational_input("Hi!"))
        self.assertTrue(is_conversational_input("Hey"))
        self.assertTrue(is_conversational_input("good morning"))
        self.assertTrue(is_conversational_input("good evening"))
        self.assertTrue(is_conversational_input("How are you?"))
        self.assertTrue(is_conversational_input("who are you"))
        self.assertTrue(is_conversational_input("thanks"))
        self.assertTrue(is_conversational_input("thank you"))
        self.assertTrue(is_conversational_input("bye"))
        self.assertTrue(is_conversational_input("goodbye"))

    def test_non_greetings_not_flagged(self):
        # Informational questions must not be flagged as conversational
        self.assertFalse(is_conversational_input("What is 2 + 2?"))
        self.assertFalse(is_conversational_input("What is Data Science?"))
        self.assertFalse(is_conversational_input("What is the capital of Georgia?"))
        self.assertFalse(is_conversational_input("Who is the president of France?"))
        self.assertFalse(is_conversational_input("Hello, can you explain neural networks?"))


if __name__ == "__main__":
    unittest.main()
