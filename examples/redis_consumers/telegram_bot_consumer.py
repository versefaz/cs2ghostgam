#!/usr/bin/env python3
"""
Telegram Bot Consumer for CS2 Betting Signals
Subscribes to Redis channels and sends betting signals to Telegram
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any

import aioredis
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CS2TelegramBot:
    def __init__(self):
        self.redis = None
        self.app = None
        self.channels = {
            'signals': os.getenv('REDIS_SIGNALS_CHANNEL', 'cs2:signals'),
            'predictions': os.getenv('REDIS_PREDICTIONS_CHANNEL', 'cs2:predictions'),
            'alerts': os.getenv('REDIS_ALERTS_CHANNEL', 'cs2:alerts')
        }
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
    async def initialize(self):
        """Initialize Redis connection and Telegram bot"""
        # Redis connection
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.redis = aioredis.from_url(redis_url)
        
        # Telegram bot
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set")
        
        self.app = Application.builder().token(bot_token).build()
        
        # Add command handlers
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("status", self.status_command))
        self.app.add_handler(CommandHandler("recent", self.recent_signals))
        self.app.add_handler(CommandHandler("help", self.help_command))
        
        # Start Redis consumers
        for channel_name, channel_key in self.channels.items():
            asyncio.create_task(self.consume_channel(channel_name, channel_key))
        
        logger.info("CS2 Telegram Bot initialized")
    
    async def consume_channel(self, channel_name: str, channel_key: str):
        """Consumer for specific Redis channel"""
        try:
            pubsub = self.redis.pubsub()
            await pubsub.subscribe(channel_key)
            
            logger.info(f"Subscribed to {channel_key}")
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        await self.handle_message(channel_name, data)
                    except Exception as e:
                        logger.error(f"Error processing message from {channel_key}: {e}")
                        
        except Exception as e:
            logger.error(f"Error in consumer for {channel_key}: {e}")
            await asyncio.sleep(5)
            asyncio.create_task(self.consume_channel(channel_name, channel_key))
    
    async def handle_message(self, channel_name: str, data: Dict[str, Any]):
        """Handle incoming Redis messages"""
        if not self.chat_id:
            return
            
        if channel_name == 'signals':
            await self.send_signal(data)
        elif channel_name == 'predictions':
            await self.send_prediction(data)
        elif channel_name == 'alerts':
            await self.send_alert(data)
    
    async def send_signal(self, signal: Dict[str, Any]):
        """Send betting signal to Telegram"""
        match = signal.get('match', {})
        action = signal.get('action', 'UNKNOWN')
        
        # Emoji based on action
        emoji = "🎯" if action == 'BET' else "⏸️"
        
        message = f"{emoji} *New Betting Signal*\n\n"
        message += f"🏆 *Match:* {match.get('team1', 'Team1')} vs {match.get('team2', 'Team2')}\n"
        message += f"📊 *Action:* {action}\n"
        message += f"🎲 *Market:* {signal.get('market', 'match_winner')}\n"
        message += f"✅ *Selection:* {signal.get('selection', 'N/A')}\n\n"
        
        message += f"💰 *Odds:* {signal.get('odds', 0):.2f}\n"
        message += f"💵 *Stake:* ${signal.get('stake', 0):.2f}\n"
        message += f"📈 *Expected Value:* {signal.get('expected_value', 0):.3f}\n\n"
        
        message += f"🎯 *Confidence:* {signal.get('confidence', 0):.1%}\n"
        message += f"⚡ *Edge:* {signal.get('edge', 0):.2%}\n"
        message += f"📊 *Kelly %:* {signal.get('kelly_fraction', 0):.2%}\n\n"
        
        timestamp = signal.get('timestamp', datetime.now().isoformat())
        message += f"🕐 *Time:* {timestamp}"
        
        # Add inline keyboard for quick actions
        keyboard = [
            [InlineKeyboardButton("📊 View Stats", callback_data=f"stats_{signal.get('match_id', '')}")],
            [InlineKeyboardButton("🔄 Refresh", callback_data="refresh")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        try:
            await self.app.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
        except Exception as e:
            logger.error(f"Error sending signal to Telegram: {e}")
    
    async def send_prediction(self, prediction: Dict[str, Any]):
        """Send match prediction to Telegram"""
        match = prediction.get('match', {})
        
        message = f"🔮 *Match Prediction*\n\n"
        message += f"🏆 *Match:* {match.get('team1', 'Team1')} vs {match.get('team2', 'Team2')}\n"
        message += f"🏅 *Predicted Winner:* {prediction.get('predicted_winner', 'N/A')}\n"
        message += f"🎯 *Confidence:* {prediction.get('confidence', 0):.1%}\n"
        message += f"🤖 *Model:* {prediction.get('model_version', 'v1.0')}\n\n"
        
        # Win probabilities
        probs = prediction.get('probabilities', {})
        if probs:
            message += "*Win Probabilities:*\n"
            for team, prob in probs.items():
                message += f"• {team}: {prob:.1%}\n"
            message += "\n"
        
        timestamp = prediction.get('timestamp', datetime.now().isoformat())
        message += f"🕐 *Time:* {timestamp}"
        
        try:
            await self.app.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"Error sending prediction to Telegram: {e}")
    
    async def send_alert(self, alert: Dict[str, Any]):
        """Send system alert to Telegram"""
        severity = alert.get('severity', 'info').upper()
        emojis = {
            'CRITICAL': '🚨',
            'ERROR': '❌',
            'WARNING': '⚠️',
            'INFO': 'ℹ️'
        }
        
        emoji = emojis.get(severity, '📢')
        
        message = f"{emoji} *{severity} Alert*\n\n"
        message += f"📝 *Message:* {alert.get('message', 'System alert')}\n"
        message += f"🔧 *Source:* {alert.get('source', 'System')}\n"
        message += f"📊 *Type:* {alert.get('alert_type', 'general')}\n"
        
        if alert.get('details'):
            details = str(alert['details'])[:500]  # Limit length
            message += f"\n*Details:*\n`{details}`\n"
        
        timestamp = alert.get('timestamp', datetime.now().isoformat())
        message += f"\n🕐 *Time:* {timestamp}"
        
        try:
            await self.app.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"Error sending alert to Telegram: {e}")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        message = (
            "🎯 *CS2 Betting System Bot*\n\n"
            "Welcome! I'll send you real-time betting signals, predictions, and alerts.\n\n"
            "*Available Commands:*\n"
            "/status - System status\n"
            "/recent - Recent signals\n"
            "/help - Show this help\n\n"
            "📊 Ready to receive signals!"
        )
        
        await update.message.reply_text(message, parse_mode='Markdown')
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        try:
            redis_info = await self.redis.info()
            
            message = (
                "📊 *CS2 Betting System Status*\n\n"
                f"🤖 Bot: Online\n"
                f"🔴 Redis: Connected\n"
                f"⏰ Uptime: {redis_info.get('uptime_in_seconds', 0)}s\n"
                f"💾 Memory: {redis_info.get('used_memory_human', 'N/A')}\n"
                f"🔗 Connections: {redis_info.get('connected_clients', 0)}\n\n"
                f"🕐 *Last Update:* {datetime.now().strftime('%H:%M:%S')}"
            )
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error getting status: {e}")
    
    async def recent_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /recent command"""
        try:
            signals_key = "cs2:signals:recent"
            signals = await self.redis.lrange(signals_key, 0, 4)  # Last 5 signals
            
            if not signals:
                await update.message.reply_text("No recent signals found.")
                return
            
            message = f"📈 *Recent {len(signals)} Signals*\n\n"
            
            for i, signal_data in enumerate(signals, 1):
                try:
                    signal = json.loads(signal_data)
                    match = signal.get('match', {})
                    
                    message += f"*{i}.* {match.get('team1', 'Team1')} vs {match.get('team2', 'Team2')}\n"
                    message += f"   Action: {signal.get('action', 'N/A')}\n"
                    message += f"   Odds: {signal.get('odds', 0):.2f}\n\n"
                    
                except json.JSONDecodeError:
                    continue
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error getting recent signals: {e}")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        message = (
            "🎯 *CS2 Betting System Bot Help*\n\n"
            "*Commands:*\n"
            "/start - Initialize bot\n"
            "/status - System status and health\n"
            "/recent - Show recent betting signals\n"
            "/help - Show this help message\n\n"
            "*Features:*\n"
            "🎯 Real-time betting signals\n"
            "🔮 Match predictions\n"
            "🚨 System alerts\n"
            "📊 Performance metrics\n\n"
            "*Support:*\n"
            "Report issues on GitHub or contact admin."
        )
        
        await update.message.reply_text(message, parse_mode='Markdown')
    
    async def run(self):
        """Run the bot"""
        await self.initialize()
        await self.app.run_polling()

async def main():
    """Main function"""
    bot = CS2TelegramBot()
    
    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Bot shutdown requested")
    except Exception as e:
        logger.error(f"Bot error: {e}")
    finally:
        if bot.redis:
            await bot.redis.close()

if __name__ == "__main__":
    asyncio.run(main())
