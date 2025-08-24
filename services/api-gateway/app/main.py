import strawberry
from strawberry.fastapi import GraphQLRouter
from fastapi import FastAPI
import grpc, orjson, httpx, os, redis
from .graphql.subscriptions import Subscription
from .hltv_socketio_client import HLTVSocketClient

@strawberry.type
class AnalyzePayload:
    market: str
    selection: str
    stake_pct: float
    ev: float
    reasoning: list[str]

@strawberry.type
class Mutation:
    @strawberry.mutation
    async def analyze(self, match_id: int) -> AnalyzePayload:
        # TODO: call prediction gRPC, read Redis features, call odds & portfolio-sim
        return AnalyzePayload(
            market="match-winner",
            selection="TeamA",
            stake_pct=0.02,
            ev=0.08,
            reasoning=["Baseline placeholder until services wired"]
        )

schema = strawberry.Schema(mutation=Mutation, subscription=Subscription)
app = FastAPI()
app.include_router(GraphQLRouter(schema), prefix="/graphql")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# HLTV Socket.io client lifecycle
client = HLTVSocketClient()

@app.on_event("startup")
async def on_startup():
    await client.start()

@app.on_event("shutdown")
async def on_shutdown():
    await client.stop()
