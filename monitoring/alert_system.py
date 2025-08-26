import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertChannel(Enum):
    DISCORD = "discord"
    TELEGRAM = "telegram"
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"


@dataclass
class Alert:
    id: str
    title: str
    message: str
    severity: AlertSeverity
    service: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None


@dataclass
class AlertRule:
    id: str
    name: str
    condition: str
    threshold: float
    severity: AlertSeverity
    channels: List[AlertChannel]
    cooldown_minutes: int = 15
    enabled: bool = True
    metadata: Dict[str, Any] = None


class AlertManager:
    """Comprehensive alert management system"""
    
    def __init__(self):
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_rules: Dict[str, AlertRule] = {}
        self.cooldowns: Dict[str, datetime] = {}
        self.channels: Dict[AlertChannel, Dict[str, Any]] = {}
        
        # Setup default alert rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default alert rules"""
        default_rules = [
            AlertRule(
                id="scraper_failure_rate",
                name="High Scraper Failure Rate",
                condition="scraper_error_rate > 0.1",
                threshold=0.1,
                severity=AlertSeverity.HIGH,
                channels=[AlertChannel.DISCORD, AlertChannel.TELEGRAM],
                cooldown_minutes=30
            ),
            AlertRule(
                id="prediction_accuracy_drop",
                name="Prediction Accuracy Drop",
                condition="prediction_accuracy < 0.5",
                threshold=0.5,
                severity=AlertSeverity.MEDIUM,
                channels=[AlertChannel.DISCORD],
                cooldown_minutes=60
            ),
            AlertRule(
                id="redis_connection_failure",
                name="Redis Connection Failure",
                condition="redis_connection_errors > 5",
                threshold=5,
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.DISCORD, AlertChannel.TELEGRAM, AlertChannel.EMAIL],
                cooldown_minutes=5
            ),
            AlertRule(
                id="high_system_cpu",
                name="High System CPU Usage",
                condition="system_cpu_usage > 85",
                threshold=85,
                severity=AlertSeverity.MEDIUM,
                channels=[AlertChannel.DISCORD],
                cooldown_minutes=20
            ),
            AlertRule(
                id="low_bankroll",
                name="Low Bankroll Warning",
                condition="bankroll_amount < 1000",
                threshold=1000,
                severity=AlertSeverity.HIGH,
                channels=[AlertChannel.DISCORD, AlertChannel.EMAIL],
                cooldown_minutes=120
            ),
            AlertRule(
                id="high_drawdown",
                name="High Drawdown Alert",
                condition="current_drawdown > 0.15",
                threshold=0.15,
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.DISCORD, AlertChannel.TELEGRAM, AlertChannel.EMAIL],
                cooldown_minutes=10
            ),
            AlertRule(
                id="queue_backup",
                name="Queue Backup",
                condition="queue_size > 1000",
                threshold=1000,
                severity=AlertSeverity.MEDIUM,
                channels=[AlertChannel.DISCORD],
                cooldown_minutes=30
            ),
            AlertRule(
                id="service_down",
                name="Service Down",
                condition="service_health == 'unhealthy'",
                threshold=0,
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.DISCORD, AlertChannel.TELEGRAM],
                cooldown_minutes=5
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.id] = rule
    
    def configure_channel(self, channel: AlertChannel, config: Dict[str, Any]):
        """Configure alert channel"""
        self.channels[channel] = config
        logger.info(f"Configured {channel.value} alert channel")
    
    async def check_conditions(self, metrics: Dict[str, Any]):
        """Check all alert conditions against current metrics"""
        for rule_id, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                if await self._evaluate_condition(rule, metrics):
                    await self._trigger_alert(rule, metrics)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule_id}: {e}")
    
    async def _evaluate_condition(self, rule: AlertRule, metrics: Dict[str, Any]) -> bool:
        """Evaluate alert condition"""
        condition = rule.condition
        
        # Simple condition evaluation
        if "scraper_error_rate >" in condition:
            error_rate = self._calculate_error_rate(metrics, 'scraper')
            return error_rate > rule.threshold
        
        elif "prediction_accuracy <" in condition:
            accuracy = metrics.get('prediction_accuracy', 1.0)
            return accuracy < rule.threshold
        
        elif "redis_connection_errors >" in condition:
            errors = metrics.get('redis_connection_errors', 0)
            return errors > rule.threshold
        
        elif "system_cpu_usage >" in condition:
            cpu_usage = metrics.get('system_cpu_usage', 0)
            return cpu_usage > rule.threshold
        
        elif "bankroll_amount <" in condition:
            bankroll = metrics.get('bankroll_amount', float('inf'))
            return bankroll < rule.threshold
        
        elif "current_drawdown >" in condition:
            drawdown = metrics.get('current_drawdown', 0)
            return drawdown > rule.threshold
        
        elif "queue_size >" in condition:
            max_queue_size = max(
                metrics.get('queue_sizes', {}).values(),
                default=0
            )
            return max_queue_size > rule.threshold
        
        elif "service_health ==" in condition:
            unhealthy_services = [
                service for service, health in metrics.get('service_health', {}).items()
                if health != 'healthy'
            ]
            return len(unhealthy_services) > 0
        
        return False
    
    def _calculate_error_rate(self, metrics: Dict[str, Any], service: str) -> float:
        """Calculate error rate for a service"""
        total_requests = metrics.get(f'{service}_total_requests', 0)
        error_requests = metrics.get(f'{service}_error_requests', 0)
        
        if total_requests == 0:
            return 0.0
        
        return error_requests / total_requests
    
    async def _trigger_alert(self, rule: AlertRule, metrics: Dict[str, Any]):
        """Trigger an alert"""
        # Check cooldown
        if rule.id in self.cooldowns:
            cooldown_end = self.cooldowns[rule.id] + timedelta(minutes=rule.cooldown_minutes)
            if datetime.now() < cooldown_end:
                return
        
        # Create alert
        alert = Alert(
            id=f"{rule.id}_{int(datetime.now().timestamp())}",
            title=rule.name,
            message=self._generate_alert_message(rule, metrics),
            severity=rule.severity,
            service=rule.metadata.get('service', 'system') if rule.metadata else 'system',
            timestamp=datetime.now(),
            metadata={'rule_id': rule.id, 'metrics_snapshot': metrics}
        )
        
        # Store alert
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        # Set cooldown
        self.cooldowns[rule.id] = datetime.now()
        
        # Send to channels
        for channel in rule.channels:
            try:
                await self._send_alert(alert, channel)
            except Exception as e:
                logger.error(f"Failed to send alert to {channel.value}: {e}")
        
        logger.warning(f"Alert triggered: {alert.title}")
    
    def _generate_alert_message(self, rule: AlertRule, metrics: Dict[str, Any]) -> str:
        """Generate alert message"""
        base_message = f"🚨 **{rule.name}**\n\n"
        
        if "scraper_error_rate" in rule.condition:
            error_rate = self._calculate_error_rate(metrics, 'scraper')
            base_message += f"Scraper error rate: {error_rate:.1%} (threshold: {rule.threshold:.1%})"
        
        elif "prediction_accuracy" in rule.condition:
            accuracy = metrics.get('prediction_accuracy', 0)
            base_message += f"Prediction accuracy: {accuracy:.1%} (threshold: {rule.threshold:.1%})"
        
        elif "redis_connection_errors" in rule.condition:
            errors = metrics.get('redis_connection_errors', 0)
            base_message += f"Redis connection errors: {errors} (threshold: {rule.threshold})"
        
        elif "system_cpu_usage" in rule.condition:
            cpu = metrics.get('system_cpu_usage', 0)
            base_message += f"System CPU usage: {cpu:.1f}% (threshold: {rule.threshold}%)"
        
        elif "bankroll_amount" in rule.condition:
            bankroll = metrics.get('bankroll_amount', 0)
            base_message += f"Bankroll amount: ${bankroll:,.2f} (threshold: ${rule.threshold:,.2f})"
        
        elif "current_drawdown" in rule.condition:
            drawdown = metrics.get('current_drawdown', 0)
            base_message += f"Current drawdown: {drawdown:.1%} (threshold: {rule.threshold:.1%})"
        
        elif "queue_size" in rule.condition:
            queue_sizes = metrics.get('queue_sizes', {})
            max_queue = max(queue_sizes.values()) if queue_sizes else 0
            base_message += f"Max queue size: {max_queue} (threshold: {rule.threshold})"
        
        elif "service_health" in rule.condition:
            unhealthy = [
                service for service, health in metrics.get('service_health', {}).items()
                if health != 'healthy'
            ]
            base_message += f"Unhealthy services: {', '.join(unhealthy)}"
        
        base_message += f"\n\n⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        base_message += f"\n🔥 Severity: {rule.severity.value.upper()}"
        
        return base_message
    
    async def _send_alert(self, alert: Alert, channel: AlertChannel):
        """Send alert to specific channel"""
        if channel not in self.channels:
            logger.warning(f"Channel {channel.value} not configured")
            return
        
        config = self.channels[channel]
        
        if channel == AlertChannel.DISCORD:
            await self._send_discord_alert(alert, config)
        elif channel == AlertChannel.TELEGRAM:
            await self._send_telegram_alert(alert, config)
        elif channel == AlertChannel.EMAIL:
            await self._send_email_alert(alert, config)
        elif channel == AlertChannel.SLACK:
            await self._send_slack_alert(alert, config)
        elif channel == AlertChannel.WEBHOOK:
            await self._send_webhook_alert(alert, config)
    
    async def _send_discord_alert(self, alert: Alert, config: Dict[str, Any]):
        """Send alert to Discord"""
        webhook_url = config.get('webhook_url')
        if not webhook_url:
            raise ValueError("Discord webhook_url not configured")
        
        # Color based on severity
        color_map = {
            AlertSeverity.LOW: 0x00ff00,      # Green
            AlertSeverity.MEDIUM: 0xffff00,   # Yellow
            AlertSeverity.HIGH: 0xff8000,     # Orange
            AlertSeverity.CRITICAL: 0xff0000  # Red
        }
        
        embed = {
            "title": alert.title,
            "description": alert.message,
            "color": color_map.get(alert.severity, 0x808080),
            "timestamp": alert.timestamp.isoformat(),
            "fields": [
                {"name": "Service", "value": alert.service, "inline": True},
                {"name": "Severity", "value": alert.severity.value.upper(), "inline": True},
                {"name": "Alert ID", "value": alert.id[:8], "inline": True}
            ]
        }
        
        payload = {"embeds": [embed]}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status != 204:
                    raise Exception(f"Discord webhook failed: {response.status}")
    
    async def _send_telegram_alert(self, alert: Alert, config: Dict[str, Any]):
        """Send alert to Telegram"""
        bot_token = config.get('bot_token')
        chat_id = config.get('chat_id')
        
        if not bot_token or not chat_id:
            raise ValueError("Telegram bot_token and chat_id must be configured")
        
        # Format message for Telegram
        message = f"🚨 *{alert.title}*\n\n"
        message += alert.message.replace('**', '*')  # Convert Discord markdown
        message += f"\n\n📊 Service: `{alert.service}`"
        message += f"\n🆔 Alert ID: `{alert.id[:8]}`"
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Telegram API failed: {response.status}")
    
    async def _send_email_alert(self, alert: Alert, config: Dict[str, Any]):
        """Send alert via email"""
        smtp_server = config.get('smtp_server')
        smtp_port = config.get('smtp_port', 587)
        username = config.get('username')
        password = config.get('password')
        to_emails = config.get('to_emails', [])
        
        if not all([smtp_server, username, password, to_emails]):
            raise ValueError("Email configuration incomplete")
        
        msg = MIMEMultipart()
        msg['From'] = username
        msg['To'] = ', '.join(to_emails)
        msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
        
        body = f"""
        Alert Details:
        
        Title: {alert.title}
        Severity: {alert.severity.value.upper()}
        Service: {alert.service}
        Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        Alert ID: {alert.id}
        
        Message:
        {alert.message}
        
        This is an automated alert from the CS2 Betting System.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email in thread pool to avoid blocking
        def send_email():
            try:
                server = smtplib.SMTP(smtp_server, smtp_port)
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
                server.quit()
            except Exception as e:
                logger.error(f"Failed to send email: {e}")
                raise
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, send_email)
    
    async def _send_slack_alert(self, alert: Alert, config: Dict[str, Any]):
        """Send alert to Slack"""
        webhook_url = config.get('webhook_url')
        if not webhook_url:
            raise ValueError("Slack webhook_url not configured")
        
        # Color based on severity
        color_map = {
            AlertSeverity.LOW: "good",
            AlertSeverity.MEDIUM: "warning", 
            AlertSeverity.HIGH: "danger",
            AlertSeverity.CRITICAL: "danger"
        }
        
        attachment = {
            "color": color_map.get(alert.severity, "warning"),
            "title": alert.title,
            "text": alert.message,
            "fields": [
                {"title": "Service", "value": alert.service, "short": True},
                {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                {"title": "Time", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "short": True}
            ],
            "footer": "CS2 Betting System",
            "ts": int(alert.timestamp.timestamp())
        }
        
        payload = {"attachments": [attachment]}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Slack webhook failed: {response.status}")
    
    async def _send_webhook_alert(self, alert: Alert, config: Dict[str, Any]):
        """Send alert to custom webhook"""
        webhook_url = config.get('url')
        if not webhook_url:
            raise ValueError("Webhook URL not configured")
        
        payload = {
            "alert": asdict(alert),
            "timestamp": alert.timestamp.isoformat()
        }
        
        headers = config.get('headers', {})
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload, headers=headers) as response:
                if response.status not in [200, 201, 202]:
                    raise Exception(f"Webhook failed: {response.status}")
    
    async def resolve_alert(self, alert_id: str, resolved_by: str = "system"):
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert {alert_id} resolved by {resolved_by}")
            
            # Send resolution notification if configured
            await self._send_resolution_notification(alert, resolved_by)
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
    
    async def _send_resolution_notification(self, alert: Alert, resolved_by: str):
        """Send alert resolution notification"""
        # Only send to Discord for now
        if AlertChannel.DISCORD in self.channels:
            config = self.channels[AlertChannel.DISCORD]
            webhook_url = config.get('webhook_url')
            
            if webhook_url:
                embed = {
                    "title": f"✅ Alert Resolved: {alert.title}",
                    "description": f"Alert has been resolved by {resolved_by}",
                    "color": 0x00ff00,  # Green
                    "timestamp": datetime.now().isoformat(),
                    "fields": [
                        {"name": "Original Alert", "value": alert.id[:8], "inline": True},
                        {"name": "Duration", "value": str(datetime.now() - alert.timestamp), "inline": True}
                    ]
                }
                
                payload = {"embeds": [embed]}
                
                try:
                    async with aiohttp.ClientSession() as session:
                        await session.post(webhook_url, json=payload)
                except Exception as e:
                    logger.error(f"Failed to send resolution notification: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff]
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics"""
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        last_7d = now - timedelta(days=7)
        
        recent_alerts = [a for a in self.alert_history if a.timestamp >= last_24h]
        weekly_alerts = [a for a in self.alert_history if a.timestamp >= last_7d]
        
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len([
                a for a in recent_alerts if a.severity == severity
            ])
        
        return {
            "active_alerts": len(self.active_alerts),
            "alerts_last_24h": len(recent_alerts),
            "alerts_last_7d": len(weekly_alerts),
            "severity_breakdown_24h": severity_counts,
            "most_common_alerts": self._get_most_common_alerts(recent_alerts),
            "avg_resolution_time": self._calculate_avg_resolution_time()
        }
    
    def _get_most_common_alerts(self, alerts: List[Alert]) -> List[Dict[str, Any]]:
        """Get most common alert types"""
        alert_counts = {}
        for alert in alerts:
            rule_id = alert.metadata.get('rule_id', 'unknown') if alert.metadata else 'unknown'
            alert_counts[rule_id] = alert_counts.get(rule_id, 0) + 1
        
        return [
            {"rule_id": rule_id, "count": count}
            for rule_id, count in sorted(alert_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
    
    def _calculate_avg_resolution_time(self) -> Optional[float]:
        """Calculate average resolution time in minutes"""
        resolved_alerts = [a for a in self.alert_history if a.resolved and a.resolved_at]
        
        if not resolved_alerts:
            return None
        
        total_time = sum(
            (alert.resolved_at - alert.timestamp).total_seconds()
            for alert in resolved_alerts
        )
        
        return total_time / len(resolved_alerts) / 60  # Convert to minutes
