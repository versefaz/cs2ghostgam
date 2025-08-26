# CS2 Betting System 🎯

A comprehensive, production-ready Counter-Strike 2 betting system with advanced machine learning, event-driven backtesting, real-time monitoring, and multi-channel alerting.

## 🚀 Features

### Core Components
- **Advanced ML Pipeline**: Feature engineering, model training, and prediction with confidence scoring
- **Event-Driven Backtesting**: Historical performance analysis with advanced metrics (Sharpe, Sortino, Calmar ratios)
- **Real-Time Data Collection**: HLTV match data, team statistics, and odds scraping
- **Risk Management**: Kelly criterion betting, bankroll management, and position sizing
- **Signal Generation**: Automated betting signals with Redis pub/sub architecture

### Infrastructure
- **Monitoring**: Prometheus metrics, Grafana dashboards, multi-channel alerting
- **Containerization**: Docker services with health checks and auto-restart
- **CI/CD**: GitHub Actions with testing, security scanning, and automated deployment
- **API Gateway**: RESTful endpoints for predictions, signals, matches, and system health

## 📋 Quick Start

### Prerequisites
- Python 3.12+
- Docker & Docker Compose
- Redis 7+
- PostgreSQL 15+

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/cs2ghostgam.git
cd cs2ghostgam
```

2. **Set up environment**
```bash
cp secrets/.env.example secrets/.env
# Edit secrets/.env with your configuration
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Start services**
```bash
# Start core services
docker-compose up -d redis postgres

# Start scraping infrastructure
docker-compose -f docker-compose.scraper.yml up -d

# Start monitoring
docker-compose -f docker-compose.monitoring.yml up -d
```

5. **Run the system**
```bash
python main.py
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   ML Pipeline   │    │  Signal Engine  │
│                 │    │                 │    │                 │
│ • HLTV Scraper  │───▶│ • Feature Eng.  │───▶│ • Risk Mgmt     │
│ • Odds Scraper  │    │ • Model Training│    │ • Kelly Calc    │
│ • Team Stats    │    │ • Backtesting   │    │ • Pub/Sub       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Redis       │    │   API Gateway   │    │   Monitoring    │
│                 │    │                 │    │                 │
│ • Predictions   │◀───│ • REST API      │    │ • Prometheus    │
│ • Signals       │    │ • Health Checks │    │ • Grafana       │
│ • Match Data    │    │ • Search        │    │ • Alerts        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📊 Monitoring & Alerting

### Grafana Dashboards
Access at `http://localhost:3000` (admin/admin)
- System overview and performance metrics
- Scraping success rates and latency
- Prediction accuracy and model performance
- Financial metrics and risk monitoring

### Alert Channels
- **Discord**: Real-time notifications with rich embeds
- **Telegram**: Mobile-friendly alerts
- **Email**: Detailed reports and summaries
- **Slack**: Team collaboration alerts

### Key Metrics
- **Scraper Health**: Success rate, response time, error count
- **Model Performance**: Accuracy, precision, recall, ROI
- **System Resources**: CPU, memory, disk usage
- **Financial**: Bankroll, drawdown, daily PnL

## 🧪 Testing

### Run Tests
```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Coverage report
pytest --cov=. --cov-report=html
```

### Smoke Tests
```bash
python scripts/e2e_smoke_test.py
```

## 🚀 Deployment

### Docker Compose (Development)
```bash
docker-compose up -d
```

### Kubernetes (Production)
```bash
kubectl apply -f infra/k8s/production/
```

### CI/CD Pipeline
- **Lint**: Code quality checks (black, flake8, isort, mypy)
- **Test**: Unit and integration tests with coverage
- **Security**: Bandit security scanning and dependency checks
- **Build**: Multi-service Docker image builds
- **Deploy**: Automated staging and production deployments

## 📈 Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Win Rate | 80%+ | Tracking |
| ROI | 15%+ | Tracking |
| Max Drawdown | <10% | Monitoring |
| Uptime | 99.9% | Monitoring |

## 🔧 Configuration

### Environment Variables
Key configuration options in `.env`:

```bash
# Core Settings
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@localhost:5432/cs2_betting
INITIAL_BANKROLL=1000.0
PAPER_TRADING=true

# Risk Management
MAX_DAILY_RISK=0.05
KELLY_FRACTION=0.25
MIN_EDGE_THRESHOLD=0.05

# Notifications
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
TELEGRAM_BOT_TOKEN=your_bot_token
```

### Model Configuration
```python
# cs2_betting_system/config/settings.py
MODEL_CONFIG = {
    'algorithm': 'RandomForest',
    'n_estimators': 100,
    'max_depth': 10,
    'feature_selection': 'auto'
}
```

## 📚 API Documentation

### Predictions
```bash
# Get current predictions
GET /api/predictions/current

# Search predictions
GET /api/predictions/search?team=navi&limit=10
```

### Signals
```bash
# Get active signals
GET /api/signals/active

# Queue status
GET /api/signals/queue/status
```

### Health
```bash
# System health
GET /api/health

# Detailed metrics
GET /api/metrics
```

## 🛠️ Development

### Project Structure
```
cs2ghostgam/
├── app/                    # Core application
├── cs2_betting_system/     # Main system components
├── ml_pipeline/           # Machine learning pipeline
├── monitoring/            # Metrics and alerting
├── services/              # Microservices
├── infra/                 # Infrastructure as code
├── tests/                 # Test suites
└── scripts/               # Utility scripts
```

### Adding New Features
1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement with tests: `pytest tests/`
3. Update documentation
4. Submit PR with CI passing

### Debugging
```bash
# View logs
docker-compose logs -f scraper
docker-compose logs -f api-gateway

# Redis inspection
redis-cli monitor

# Database queries
psql $DATABASE_URL
```

## 🔒 Security

### Best Practices
- Environment variables for secrets
- API key rotation
- Rate limiting on endpoints
- Input validation and sanitization
- Regular security scans with bandit

### Monitoring
- Failed authentication attempts
- Unusual betting patterns
- System resource anomalies
- Network intrusion detection

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure CI passes
5. Submit pull request

### Code Style
- Black formatting: `black .`
- Import sorting: `isort .`
- Type hints: `mypy .`
- Linting: `flake8 .`

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🆘 Support

### Common Issues
- **Selenium WebDriver**: Ensure ChromeDriver version matches Chrome
- **Redis Connection**: Check Redis server status and credentials
- **Model Loading**: Verify model file exists at configured path

### Getting Help
- Create GitHub issue with detailed description
- Check logs for error messages
- Review monitoring dashboards for system health

### Performance Tuning
- Adjust `MAX_CONCURRENT_SCRAPES` for scraping load
- Tune Redis connection pool size
- Configure appropriate resource limits in Kubernetes

---

**⚠️ Disclaimer**: This system is for educational and research purposes. Always comply with local gambling laws and betting site terms of service. Use at your own risk.
