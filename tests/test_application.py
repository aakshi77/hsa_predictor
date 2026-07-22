import unittest

from application import app


class ApplicationTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_predict_returns_expected_keys(self):
        payload = {
            "ageBracket": "25-34",
            "enrollingAs": "Individual",
            "chronicCondition": "No",
            "region": "northwest"
        }

        response = self.client.post('/predict', json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('predictionRange', data)
        self.assertIn('recommendation', data)
        self.assertIn('taxSavings', data)


if __name__ == '__main__':
    unittest.main()
