from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(10, 40)  # Simula que los usuarios esperan entre 10-40 segundos

    @task
    def load_page_1(self):
        self.client.get("/time_trial")  # Visita la página /time_trial

    @task
    def load_page_2(self):
        self.client.get("/play")  # Visita la página /time_trial
    
    @task
    def load_page_3(self):
        self.client.get("/leaderboards?userID=77888a5c-c1e8-4f8f-a11f-ae9b21ece3ff")  # Visita la página /time_trial
    
    @task
    def load_page_4(self):
        self.client.get("/tutorial")  # Visita la página /time_trial
  
    @task
    def load_page_5(self):
        self.client.get("/display_level?level=28&n=4")  # Visita la página /time_trial

    @task
    def load_page_6(self):
        self.client.get("/leaderboard?totaltime=695&userID=77888a5c-c1e8-4f8f-a11f-ae9b21ece3ff&n=4")  # Visita la página /time_trial