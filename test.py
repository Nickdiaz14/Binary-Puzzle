from locust import HttpUser, task, between
import random

class UserBehavior(HttpUser):
    host = "http://127.0.0.1:5000"  # Dirección de tu aplicación Flask
    wait_time = between(5, 10)  # Espera entre tareas

    @task
    def send_number(self):
        number = random.randint(0, 10000)  # Genera un número aleatorio
        payload = {"num": number, "game": "1"}  # Ajuste a los datos esperados

        with self.client.post("/games", json=payload, headers={"Content-Type": "application/json"}, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Error {response.status_code}: {response.text}")

