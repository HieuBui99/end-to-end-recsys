import time
from locust import HttpUser, task, between

test_data = [1, 2]
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}

class BentoUser(HttpUser):
    wait_time = between(0.01, 2)

    @task
    def health_check(self):
        self.client.post("predict", headers=headers, json=test_data)
