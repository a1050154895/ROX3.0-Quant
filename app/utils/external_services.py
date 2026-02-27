import httpx

class ClawFeedClient:
    def __init__(self, base_url="http://localhost:8767/api"):
        self.base_url = base_url
        self.client = httpx.AsyncClient()
    
    async def get_digests(self, type="daily", limit=20):
        response = await self.client.get(f"{self.base_url}/digests", params={"type": type, "limit": limit})
        return response.json()
    
    async def get_digest(self, digest_id):
        response = await self.client.get(f"{self.base_url}/digests/{digest_id}")
        return response.json()

class OpenClawClient:
    def __init__(self, base_url="http://localhost:3000/api"):
        self.base_url = base_url
        self.client = httpx.AsyncClient()
    
    async def get_skills(self):
        response = await self.client.get(f"{self.base_url}/skills")
        return response.json()
    
    async def execute_skill(self, skill_id, params):
        response = await self.client.post(f"{self.base_url}/skills/{skill_id}/execute", json=params)
        return response.json()
    
    async def chat(self, message):
        response = await self.client.post(f"{self.base_url}/chat", json={"message": message})
        return response.json()
