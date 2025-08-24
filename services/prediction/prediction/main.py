import grpc
from concurrent import futures
import orjson
import redis
import joblib
import os
from libs.protobuf.gen import analytics_pb2 as pb2
from libs.protobuf.gen import analytics_pb2_grpc as pb2_grpc

MODEL_PATH = os.getenv("MODEL_PATH", "models/1/model.pkl")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")

class Predictor(pb2_grpc.PredictorServicer):
    def __init__(self):
        self.model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
        self.rds = redis.Redis(host=REDIS_HOST)

    def Predict(self, request, context):
        key = f"feature:{request.value}"
        raw = self.rds.get(key)
        if not raw:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("feature not found")
            return pb2.Prediction(prob=0.0, shap=[])
        vec = orjson.loads(raw)
        xs = [list(vec.values())]
        if self.model is None:
            # placeholder probability for dev
            prob = 0.5
            shap_vals = []
        else:
            prob = float(self.model.predict_proba(xs)[0][1])
            shap_vals = []  # compute later to avoid heavy dep at runtime
        return pb2.Prediction(prob=prob, shap=shap_vals[:5])

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb2_grpc.add_PredictorServicer_to_server(Predictor(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
